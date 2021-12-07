//
//  Occupancy.swift
//  testapp6
//
//  Created by DAIKI ONODERA on 2021/12/03.
//

import Foundation
import CoreML
import Vision

class Occupancy{
    var reduction:Float!
    var min_scale_reduced:Float!
    var occupancy: [MLMultiArray]!
    
    init(shape:[Int], reduction:Float, min_scale:Float){
        self.reduction = reduction
        let min_scale = reduction
        self.min_scale_reduced = min_scale / reduction
        let field_shape = [Int(Float(shape[1]) / reduction) + 1, Int(Float(shape[2]) / reduction) + 1] as [NSNumber]
        let shape_0 = shape[0]
        let shape_1 = Int(Float(shape[1]) / reduction) + 1
        let shape_2 = Int(Float(shape[2]) / reduction) + 1
        for i in 0..<shape_0{
            guard let ar = try? MLMultiArray(shape: field_shape, dataType: .float32) else{
                fatalError("fatalError at MLMultiArray.")
            }
            self.occupancy.append(ar)
        }
    }
    
    func scalar_square_set(field: MLMultiArray, x:Float, y:Float, sigma:Float){
        let x = x/self.reduction
        let y = y/self.reduction
        let sigma = max(1.0, sigma/self.reduction)
        let minx = self.clip(v: (x - sigma), minv: 0, maxv: field.shape[1].intValue - 1)
        let miny = self.clip(v: (y - sigma), minv: 0, maxv: field.shape[0].intValue - 1)
        let maxx = self.clip(v: (x + sigma), minv: minx+1, maxv: field.shape[1].intValue)
        let maxy = self.clip(v: (y + sigma), minv: miny+1, maxv: field.shape[0].intValue)
        
        for i in minx..<maxx{
            for j in miny..<maxy{
                let mlindex = [i, j] as [NSNumber]
                field[mlindex] = 1
            }
        }
    }

    func set(f:Int, x:Float, y:Float, sigma:Float){
        scalar_square_set(field: self.occupancy[f], x: x, y: y, sigma: sigma)
    }
    func clip(v:Float, minv:Int, maxv:Int)->Int{
        return Int(max(Float(minv), min(Float(maxv), v)))
    }
}
