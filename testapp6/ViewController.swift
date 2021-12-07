//
//  ViewController.swift
//  testapp6
//
//  Created by DAIKI ONODERA on 2021/11/24.
//

import UIKit
import Vision
import AVFoundation


class ViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    public let fps: Int64 = 10
    public var frameCounter:Int64 = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        var frameForTimes = [NSValue]()
        
        let imageprocessing = ImageProcessing()
        imageprocessing.setupModel()

        guard let movieURL = Bundle.main.url(forResource: "trimed_fall", withExtension: "mp4") else {
                return
        }
        let asset = AVURLAsset(url: movieURL)
        guard let track = AVURLAsset(url: movieURL).tracks(withMediaType: AVMediaType.video).first else { return }
        let size = track.naturalSize.applying(track.preferredTransform)
        
        let assetImageGenerator = AVAssetImageGenerator(asset: asset)
        assetImageGenerator.appliesPreferredTrackTransform = true
        assetImageGenerator.apertureMode = AVAssetImageGenerator.ApertureMode.encodedPixels
        assetImageGenerator.requestedTimeToleranceAfter = CMTime.zero
        assetImageGenerator.requestedTimeToleranceBefore = CMTime.zero
        
        let totalSeconds:Int64 = Int64(floor(asset.duration.seconds))
        let sampleCounts = self.fps * totalSeconds
        
        for i in 0 ..< sampleCounts {
            let cmTime = CMTimeMake(value: Int64(i), timescale: Int32(self.fps))
            frameForTimes.append(NSValue(time: cmTime))
        }

        let tempDir = FileManager.default.temporaryDirectory
        let previewURL = tempDir.appendingPathComponent("preview.mp4")
        let fileManeger = FileManager.default
        if fileManeger.fileExists(atPath: previewURL.path) {
            try! fileManeger.removeItem(at: previewURL)
        }

        guard let videoWriter = try? AVAssetWriter(outputURL: previewURL, fileType: AVFileType.mp4) else {
            return
        }

        let outputSettings: [String : Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: size.width,
            AVVideoHeightKey: size.height
        ]

        let writerInput = AVAssetWriterInput(mediaType: AVMediaType.video, outputSettings: outputSettings)
        writerInput.expectsMediaDataInRealTime = true
        videoWriter.add(writerInput)

        func testFunc(){
            writerInput.markAsFinished()
            videoWriter.endSession(atSourceTime: CMTimeMake(value: self.frameCounter, timescale:Int32(self.fps)))
            videoWriter.finishWriting {
                print("Finish writing!")
            }
        }
        
        let sourcePixelBufferAttributes: [String:Any] = [
            AVVideoCodecKey: Int(kCVPixelFormatType_32ARGB),
            AVVideoWidthKey: size.width,
            AVVideoHeightKey: size.height
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes)

        if (!videoWriter.startWriting()) {
            print("Failed to start writing.")
            return
        }
        videoWriter.startSession(atSourceTime: CMTime.zero)
        
        assetImageGenerator.generateCGImagesAsynchronously(forTimes: frameForTimes, completionHandler: {requestedTime, image, actualTime, result, error in
                if let image = image {
                    imageprocessing.inferece(image: image)
                }
        })
    }
}
