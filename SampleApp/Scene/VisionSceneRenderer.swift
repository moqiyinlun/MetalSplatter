#if os(visionOS)

import CompositorServices
import Metal
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import Spatial
import SwiftUI

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

class VisionSceneRenderer {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "CompsitorServicesSceneRenderer")

    let layerRenderer: LayerRenderer
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider

    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!

        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()
    }

    func load(_ model: ModelIdentifier?) throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url):
            let splat = try SplatRenderer(device: device,
                                          colorFormat: layerRenderer.configuration.colorFormat,
                                          depthFormat: layerRenderer.configuration.depthFormat,
                                          stencilFormat: .invalid,
                                          sampleCount: 1,
                                          maxViewCount: layerRenderer.properties.viewCount,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            // Store depth on visionOS because it's used for reprojection by the frame interpolator, since we don't hit a solid 90fps
            splat.storeDepth = true
            try splat.read(from: url)
            modelRenderer = splat
        case .sampleBox:
            modelRenderer = try! SampleBoxRenderer(device: device,
                                                   colorFormat: layerRenderer.configuration.colorFormat,
                                                   depthFormat: layerRenderer.configuration.depthFormat,
                                                   stencilFormat: .invalid,
                                                   sampleCount: 1,
                                                   maxViewCount: layerRenderer.properties.viewCount,
                                                   maxSimultaneousRenders: Constants.maxSimultaneousRenders)
        case .none:
            break
        }
    }

    func startRenderLoop() {
        Task {
            do {
                try await arSession.run([worldTracking])
            } catch {
                fatalError("Failed to initialize ARSession")
            }

            let renderThread = Thread {
                self.renderLoop()
            }
            renderThread.name = "Render Thread"
            renderThread.start()
        }
    }

    private func viewports(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) -> [ModelRendererViewportDescriptor] {
        let rotationMatrix = matrix4x4_rotation(radians: Float(rotation.radians),
                                                axis: Constants.rotationAxis)
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
        // Turn common 3D GS PLY files rightside-up. This isn't generally meaningful, it just
        // happens to be a useful default for the most common datasets at the moment.
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let simdDeviceAnchor = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        return drawable.views.map { view in
            let userViewpointMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = ProjectiveTransform3D(leftTangent: Double(view.tangents[0]),
                                                         rightTangent: Double(view.tangents[1]),
                                                         topTangent: Double(view.tangents[2]),
                                                         bottomTangent: Double(view.tangents[3]),
                                                         nearZ: Double(drawable.depthRange.y),
                                                         farZ: Double(drawable.depthRange.x),
                                                         reverseZ: true)
            let screenSize = SIMD2(x: Int(view.textureMap.viewport.width),
                                   y: Int(view.textureMap.viewport.height))
            return ModelRendererViewportDescriptor(viewport: view.textureMap.viewport,
                                                   projectionMatrix: .init(projectionMatrix),
                                                   viewMatrix: userViewpointMatrix * translationMatrix * rotationMatrix * commonUpCalibration,
                                                   screenSize: screenSize)
        }
    }

    private func updateRotation() {
        let now = Date()
        defer {
            lastRotationUpdateTimestamp = now
        }

        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }

        frame.startUpdate()
        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        guard let drawable = frame.queryDrawable() else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        updateRotation()

        let viewports = self.viewports(drawable: drawable, deviceAnchor: deviceAnchor)
        let tempTextureDescriptor = MTLTextureDescriptor()
        tempTextureDescriptor.textureType = .type2DArray 
        tempTextureDescriptor.pixelFormat = drawable.colorTextures[0].pixelFormat
        tempTextureDescriptor.width = drawable.colorTextures[0].width
        tempTextureDescriptor.height = drawable.colorTextures[0].height
        tempTextureDescriptor.arrayLength = 2 
        tempTextureDescriptor.usage = [.shaderWrite, .shaderRead]  
        tempTextureDescriptor.storageMode = .private  
        tempTextureDescriptor.mipmapLevelCount = 1 
        guard let tempTexture = device.makeTexture(descriptor: tempTextureDescriptor) else {
            fatalError("Failed to create temporary texture array")
        }
        modelRenderer?.render(viewports: viewports,
                              colorTexture: drawable.colorTextures[0],
                              colorStoreAction: .store,
                              depthTexture: drawable.depthTextures[0],
                              stencilTexture: nil,
                              rasterizationRateMap: drawable.rasterizationRateMaps.first,
                              renderTargetArrayLength: layerRenderer.configuration.layout == .layered ? drawable.views.count : 1,
                              to: commandBuffer)
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create compute command encoder")
        }
        guard let computePipelineState = computePipelineState else {
            fatalError("computePipelineState is nil")
        }
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setTexture(drawable.colorTextures[0], index: 0)
        computeEncoder.setTexture(tempTexture, index: 1)

        let width = computePipelineState.threadExecutionWidth
        let height = computePipelineState.maxTotalThreadsPerThreadgroup / width
        let threadsPerThreadgroup = MTLSize(width: width, height: height, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (drawable.colorTextures[0].width + width - 1) / width,
                                          height: (drawable.colorTextures[0].height + height - 1) / height,
                                          depth: 1)

        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            fatalError("Failed to create blit command encoder")
        }

        for layer in 0..<2 {
            blitEncoder.copy(from: tempTexture,
                                sourceSlice: layer,
                                sourceLevel: 0,
                                sourceOrigin: MTLOriginMake(0, 0, 0),
                                sourceSize: MTLSizeMake(tempTexture.width, tempTexture.height, 1),
                                to: drawable.colorTextures[0],
                                destinationSlice: layer,
                                destinationLevel: 0,
                                destinationOrigin: MTLOriginMake(0, 0, 0))
        }

        blitEncoder.endEncoding()
        drawable.encodePresent(commandBuffer: commandBuffer)

        commandBuffer.commit()

        frame.endSubmission()
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                Self.log.warning("Layer is invalidated")
                return
            } else if layerRenderer.state == .paused {
                layerRenderer.waitUntilRunning()
                continue
            } else {
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }
}

#endif // os(visionOS)

