/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The host app renderer.
*/

import Foundation
import Metal
import MetalKit
import ARKit

protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var depthStencilPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}

// The max number of command buffers in flight
let kMaxBuffersInFlight: Int = 3

// The 16 byte aligned size of our uniform structures
let kAlignedSharedUniformsSize: Int = (MemoryLayout<SharedUniforms>.size & ~0xFF) + 0x100

// Vertex data for an image plane
let kImagePlaneVertexData: [Float] = [
    -1.0, -1.0, 0.0, 1.0,
    1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 0.0
]

class Renderer {
    let session: ARSession
    let matteGenerator: ARMatteGenerator
    let device: MTLDevice
    let inFlightSemaphore = DispatchSemaphore(value: kMaxBuffersInFlight)
    var renderDestination: RenderDestinationProvider
    
    // Metal objects
    var commandQueue: MTLCommandQueue!
    var sharedUniformBuffer: MTLBuffer!
    var imagePlaneVertexBuffer: MTLBuffer!
    var scenePlaneVertexBuffer: MTLBuffer!
    var compositePipelineState: MTLRenderPipelineState!
    var compositeDepthState: MTLDepthStencilState!

    var capturedImageTextureY: CVMetalTexture?
    var capturedImageTextureCbCr: CVMetalTexture?

    // Matting textures to be filled by ARMattingGenerator
    var alphaTexture: MTLTexture?
    var dilatedDepthTexture: MTLTexture?

    // Temporary render targets for 3d scene
    var sceneColorTexture: MTLTexture!
    var sceneDepthTexture: MTLTexture!
    
    // Captured image texture cache
    var capturedImageTextureCache: CVMetalTextureCache!
    
    // Used to determine _uniformBufferStride each frame.
    //   This is the current frame number modulo kMaxBuffersInFlight
    var uniformBufferIndex: Int = 0
    
    // Offset within _sharedUniformBuffer to set for the current frame
    var sharedUniformBufferOffset: Int = 0
    
    // The current viewport size
    var viewportSize: CGSize = CGSize()
    
    // Flag for viewport size changes
    var viewportSizeDidChange: Bool = false
    
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        matteGenerator = ARMatteGenerator(device: device, matteResolution: .half)
        loadMetal()
    }
    
    func drawRectResized(size: CGSize) {
        viewportSize = size
        viewportSizeDidChange = true
    }
    
    func update() {
        // Wait to ensure only kMaxBuffersInFlight are getting proccessed by any stage in the Metal
        //   pipeline (App, Metal, Drivers, GPU, etc)
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        // Create a new command buffer for each renderpass to the current drawable
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            commandBuffer.label = "MyCommand"
            
            // Add completion hander which signal _inFlightSemaphore when Metal and the GPU has fully
            //   finished proccssing the commands we're encoding this frame.  This indicates when the
            //   dynamic buffers, that we're writing to this frame, will no longer be needed by Metal
            //   and the GPU.
            // Retain our CVMetalTextures for the duration of the rendering cycle. The MTLTextures
            //   we use from the CVMetalTextures are not valid unless their parent CVMetalTextures
            //   are retained. Since we may release our CVMetalTexture ivars during the rendering
            //   cycle, we must retain them separately here.
            var textures = [capturedImageTextureY, capturedImageTextureCbCr]
            commandBuffer.addCompletedHandler { [weak self] commandBuffer in
                if let strongSelf = self {
                    strongSelf.inFlightSemaphore.signal()
                }
                textures.removeAll()
            }
            
            updateBufferStates()
            updateGameState()
            updateMatteTextures(commandBuffer: commandBuffer)
            
            if let renderPassDescriptor = renderDestination.currentRenderPassDescriptor, let currentDrawable = renderDestination.currentDrawable {

                // Setup offscreen render pass for later compositing
                guard let sceneRenderDescriptor = renderPassDescriptor.copy() as? MTLRenderPassDescriptor else {
                    fatalError("Unable to create a render pass descriptor.")
                }
                sceneRenderDescriptor.colorAttachments[0].texture = sceneColorTexture
                sceneRenderDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
                sceneRenderDescriptor.colorAttachments[0].loadAction = .clear
                sceneRenderDescriptor.colorAttachments[0].storeAction = .store

                sceneRenderDescriptor.depthAttachment.texture = sceneDepthTexture
                sceneRenderDescriptor.depthAttachment.clearDepth = 1.0
                sceneRenderDescriptor.depthAttachment.loadAction = .clear
                sceneRenderDescriptor.depthAttachment.storeAction = .store


                // Perform final composite pass
                // Here we take the targets we just rendered camera and scene into and decide whether
                // to hide or show virtual content.
                if let compositeRenderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {

                    compositeRenderEncoder.label = "MyCompositeRenderEncoder"

                    // Composite images to final render targets
                    compositeImagesWithEncoder(renderEncoder: compositeRenderEncoder)

                    // We're done encoding commands
                    compositeRenderEncoder.endEncoding()
                }
                
                // Schedule a present once the framebuffer is complete using the current drawable
                commandBuffer.present(currentDrawable)
            }
            
            // Finalize rendering here & push the command buffer to the GPU
            commandBuffer.commit()
        }
    }
    
    // MARK: - Private
    func loadMetal() {
        // Create and load our basic Metal state objects
        
        // Set the default formats needed to render
        renderDestination.depthStencilPixelFormat = .depth32Float
        renderDestination.colorPixelFormat = .bgra8Unorm
        renderDestination.sampleCount = 1
        
        // Calculate our uniform buffer sizes. We allocate kMaxBuffersInFlight instances for uniform
        //   storage in a single buffer. This allows us to update uniforms in a ring (i.e. triple
        //   buffer the uniforms) so that the GPU reads from one slot in the ring wil the CPU writes
        //   to another. Anchor uniforms should be specified with a max instance count for instancing.
        //   Also uniform storage must be aligned (to 256 bytes) to meet the requirements to be an
        //   argument in the constant address space of our shading functions.
        let sharedUniformBufferSize = kAlignedSharedUniformsSize * kMaxBuffersInFlight
        
        // Create and allocate our uniform buffer objects. Indicate shared storage so that both the
        //   CPU can access the buffer
        sharedUniformBuffer = device.makeBuffer(length: sharedUniformBufferSize, options: .storageModeShared)
        sharedUniformBuffer.label = "SharedUniformBuffer"
        
        // Create a vertex buffer with our image plane vertex data.
        let imagePlaneVertexDataCount = kImagePlaneVertexData.count * MemoryLayout<Float>.size
        imagePlaneVertexBuffer = device.makeBuffer(bytes: kImagePlaneVertexData, length: imagePlaneVertexDataCount, options: [])
        imagePlaneVertexBuffer.label = "ImagePlaneVertexBuffer"

        scenePlaneVertexBuffer = device.makeBuffer(bytes: kImagePlaneVertexData, length: imagePlaneVertexDataCount, options: [])
        scenePlaneVertexBuffer.label = "ScenePlaneVertexBuffer"
        
        // Load all the shader files with a metal file extension in the project
        let defaultLibrary = device.makeDefaultLibrary()!

        // Create a vertex descriptor for our image plane vertex buffer
        let imagePlaneVertexDescriptor = MTLVertexDescriptor()
        
        // Positions.
        imagePlaneVertexDescriptor.attributes[0].format = .float2
        imagePlaneVertexDescriptor.attributes[0].offset = 0
        imagePlaneVertexDescriptor.attributes[0].bufferIndex = Int(kBufferIndexMeshPositions.rawValue)
        
        // Texture coordinates.
        imagePlaneVertexDescriptor.attributes[1].format = .float2
        imagePlaneVertexDescriptor.attributes[1].offset = 8
        imagePlaneVertexDescriptor.attributes[1].bufferIndex = Int(kBufferIndexMeshPositions.rawValue)
        
        // Buffer Layout
        imagePlaneVertexDescriptor.layouts[0].stride = 16
        imagePlaneVertexDescriptor.layouts[0].stepRate = 1
        imagePlaneVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        // Create captured image texture cache
        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        capturedImageTextureCache = textureCache

        // Create render targets for offscreen camera image and scene render
        let width = renderDestination.currentDrawable!.texture.width
        let height = renderDestination.currentDrawable!.texture.height

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: renderDestination.colorPixelFormat,
                                                                 width: width, height: height, mipmapped: false)
        colorDesc.usage = MTLTextureUsage(rawValue: MTLTextureUsage.renderTarget.rawValue | MTLTextureUsage.shaderRead.rawValue)
        sceneColorTexture = device.makeTexture(descriptor: colorDesc)

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: renderDestination.depthStencilPixelFormat,
                                                                 width: width, height: height, mipmapped: false)
        depthDesc.usage = MTLTextureUsage(rawValue: MTLTextureUsage.renderTarget.rawValue | MTLTextureUsage.shaderRead.rawValue)
        sceneDepthTexture = device.makeTexture(descriptor: depthDesc)

        // Create composite pipeline
        let compositeImageVertexFunction = defaultLibrary.makeFunction(name: "compositeImageVertexTransform")!
        let compositeImageFragmentFunction = defaultLibrary.makeFunction(name: "compositeImageFragmentShader")!

        let compositePipelineStateDescriptor = MTLRenderPipelineDescriptor()
        compositePipelineStateDescriptor.label = "MyCompositePipeline"
        compositePipelineStateDescriptor.sampleCount = renderDestination.sampleCount
        compositePipelineStateDescriptor.vertexFunction = compositeImageVertexFunction
        compositePipelineStateDescriptor.fragmentFunction = compositeImageFragmentFunction
        compositePipelineStateDescriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        compositePipelineStateDescriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat

        do {
            try compositePipelineState = device.makeRenderPipelineState(descriptor: compositePipelineStateDescriptor)
        } catch let error {
            print("Failed to create composite pipeline state, error \(error)")
        }

        let compositeDepthStateDescriptor = MTLDepthStencilDescriptor()
        compositeDepthStateDescriptor.depthCompareFunction = .always
        compositeDepthStateDescriptor.isDepthWriteEnabled = false
        compositeDepthState = device.makeDepthStencilState(descriptor: compositeDepthStateDescriptor)

        // Create the command queue
        commandQueue = device.makeCommandQueue()
    }
    
    func updateBufferStates() {
        // Update the location(s) to which we'll write to in our dynamically changing Metal buffers for
        //   the current frame (i.e. update our slot in the ring buffer used for the current frame)
        
        uniformBufferIndex = (uniformBufferIndex + 1) % kMaxBuffersInFlight
        
        sharedUniformBufferOffset = kAlignedSharedUniformsSize * uniformBufferIndex
    }
    
    func updateGameState() {
        // Update any game state
        
        guard let currentFrame = session.currentFrame else {
            return
        }
        
//        updateSharedUniforms(frame: currentFrame)
//        updateAnchors(frame: currentFrame)
        updateCapturedImageTextures(frame: currentFrame)
        
        if viewportSizeDidChange {
            viewportSizeDidChange = false
            
            updateImagePlane(frame: currentFrame)
        }
    }
    
    func updateCapturedImageTextures(frame: ARFrame) {
        // Create two textures (Y and CbCr) from the provided frame's captured image
        let pixelBuffer = frame.capturedImage
        
        if CVPixelBufferGetPlaneCount(pixelBuffer) < 2 {
            return
        }
        
        capturedImageTextureY = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0)
        capturedImageTextureCbCr = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1)
    }

    func updateMatteTextures(commandBuffer: MTLCommandBuffer) {
        guard let currentFrame = session.currentFrame else {
            return
        }

        alphaTexture = matteGenerator.generateMatte(from: currentFrame, commandBuffer: commandBuffer)

        dilatedDepthTexture = matteGenerator.generateDilatedDepth(from: currentFrame, commandBuffer: commandBuffer)
    }
    
    func createTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, capturedImageTextureCache, pixelBuffer, nil, pixelFormat,
                                                               width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }
        
        return texture
    }
    
    func updateImagePlane(frame: ARFrame) {
        // Update the texture coordinates of our image plane to aspect fill the viewport
        let displayToCameraTransform = frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()

        let vertexData = imagePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
        let compositeVertexData = scenePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0...3 {
            let textureCoordIndex = 4 * index + 2
            let textureCoord = CGPoint(x: CGFloat(kImagePlaneVertexData[textureCoordIndex]), y: CGFloat(kImagePlaneVertexData[textureCoordIndex + 1]))
            let transformedCoord = textureCoord.applying(displayToCameraTransform)
            vertexData[textureCoordIndex] = Float(transformedCoord.x)
            vertexData[textureCoordIndex + 1] = Float(transformedCoord.y)

            compositeVertexData[textureCoordIndex] = kImagePlaneVertexData[textureCoordIndex]
            compositeVertexData[textureCoordIndex + 1] = kImagePlaneVertexData[textureCoordIndex + 1]
        }
    }
    
    func compositeImagesWithEncoder(renderEncoder: MTLRenderCommandEncoder) {
        guard let textureY = capturedImageTextureY, let textureCbCr = capturedImageTextureCbCr else {
            return
        }

        // Push a debug group allowing us to identify render commands in the GPU Frame Capture tool
        renderEncoder.pushDebugGroup("CompositePass")

        // Set render command encoder state
        renderEncoder.setCullMode(.none)
        renderEncoder.setRenderPipelineState(compositePipelineState)
        renderEncoder.setDepthStencilState(compositeDepthState)

        // Setup plane vertex buffers
        renderEncoder.setVertexBuffer(imagePlaneVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(scenePlaneVertexBuffer, offset: 0, index: 1)

        // Setup textures for the composite fragment shader
        renderEncoder.setFragmentBuffer(sharedUniformBuffer, offset: sharedUniformBufferOffset, index: Int(kBufferIndexSharedUniforms.rawValue))
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(textureY), index: 0)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(textureCbCr), index: 1)
        renderEncoder.setFragmentTexture(sceneColorTexture, index: 2)
        renderEncoder.setFragmentTexture(sceneDepthTexture, index: 3)
        renderEncoder.setFragmentTexture(alphaTexture, index: 4)
        renderEncoder.setFragmentTexture(dilatedDepthTexture, index: 5)

        // Draw final quad to display
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.popDebugGroup()
    }
}
