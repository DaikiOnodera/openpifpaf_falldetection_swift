//
//  ImageProcessing.swift
//  testapp6
//
//  Created by DAIKI ONODERA on 2021/12/02.
//

import Foundation
import Vision

class ImageProcessing{
    private var request:VNRequest!
    private var decoder: Decoder!

    init(){
        self.decoder = Decoder()
    }
    
    func inferece(image: CGImage){
        let imageRequestHandler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try imageRequestHandler.perform([self.request])
        } catch {
            print("Request cannot perform because \(error)")
        }
        guard let results = self.request.results as? [VNCoreMLFeatureValueObservation] else {
                        fatalError("Model failed to process image")
        }
        let cif_head = results[0].featureValue.multiArrayValue!
        let caf_head = results[1].featureValue.multiArrayValue!
        self.decoder.detectKeypoints(cif_head: cif_head, caf_head: caf_head)
        print(decoder.anns)
    }
    
    func setupModel(){
        do {
            guard let modelURL = Bundle.main.url(forResource: "openpifpaf-resnet50", withExtension: "mlmodelc") else {
                fatalError("Model file is missing")
            }
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel)
            self.request = objectRecognition
        } catch let error as NSError {
            fatalError("Model loading went wrong: \(error)")
        }
    }
}
