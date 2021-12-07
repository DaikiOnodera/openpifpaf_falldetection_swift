//
//  Decoder.swift
//  testapp6
//
//  Created by DAIKI ONODERA on 2021/12/02.
//

import Foundation
import CoreML

struct CafScored{
    var caf_f: [Int: [[Float]]] = [Int: [[Float]]]()
    
    init(n_skeleton: Int){
        for i in 0..<n_skeleton{
            self.caf_f[i] = [[Float]]()
        }
    }
    public func getCount(i: Int)->Int{
        return self.caf_f[i]!.count
    }
}

struct Frontier{
    let score:Float
    let new_xysv:[Float]
    let start_i:Int
    let end_i:Int
    var isEmpty:Bool = false

    init(score: Float, new_xysv:[Float], start_i:Int, end_i:Int){
        self.score = score
        self.new_xysv = new_xysv
        self.start_i = start_i
        self.end_i = end_i
    }
}

struct Ann{
    let v:Float
    let x:Float
    let y:Float
    let s:Float
    
    init(v:Float, x:Float, y:Float, s:Float){
        self.v = v
        self.x = x
        self.y = y
        self.s = s
    }
}

class Decoder{
    
    public var seeds: [[Float]] = [[Float]]()
    public var cafScored: CafScored!
    public var caf_forward: [Int: CafScored] = [Int: CafScored]()
    public var by_source: [Int: [Int: [Any]]] = [Int: [Int: [Any]]]()
    public var anns:[Int: Ann] = [Int: Ann]()
    public var in_frontier: [(Int, Int)] = [(Int, Int)]()
    public var frontier: [Frontier] = [Frontier]()
    public let skeleton_m1 = [[15, 13],
                            [13, 11],
                            [16, 14],
                            [14, 12],
                            [11, 12],
                            [ 5, 11],
                            [ 6, 12],
                            [ 5,  6],
                            [ 5,  7],
                            [ 6,  8],
                            [ 7,  9],
                            [ 8, 10],
                            [ 1,  2],
                            [ 0,  1],
                            [ 0,  2],
                            [ 1,  3],
                            [ 2,  4],
                            [ 3,  5],
                            [ 4,  6]]


    init(){
        for (i, ele) in self.skeleton_m1.enumerated(){
            // Initialize directed joints
            let sub_forward = [ele[1]: [i, true]]
            self.by_source.updateValue(sub_forward, forKey: ele[0])
        }
        // Initialize cafScored
        self.cafScored = CafScored(n_skeleton: self.skeleton_m1.count)
    }

    func detectKeypoints(cif_head: MLMultiArray, caf_head: MLMultiArray){
        self.cif_fill(cif_head: cif_head)
        self.caf_fill(caf_head: caf_head)
        
        let vfxys = self.seeds[0]
        let index_f = Int(vfxys[1])
        self.anns[index_f] = Ann(v: vfxys[0], x: vfxys[2], y: vfxys[3], s: vfxys[4])
        self._grow(vfxys: vfxys)
    }

    func grow_connection_blend(caf_f:[[Float]], x:Float, y:Float, scale:Float)->[Float]{
        let sigma_filter = 2.0 * scale
        let sigma2 = 0.25 * scale * scale
        let n_candidate = caf_f.count
        var score_1_i:Int = 0
        var score_2_i:Int = 0
        var score_1:Float = 0.0
        var score_2:Float = 0.0
        
        for i in 0..<n_candidate{
            guard caf_f[i][1] > (x - sigma_filter) && caf_f[i][1] < (x + sigma_filter) &&
                  caf_f[i][2] > (y - sigma_filter) && caf_f[i][2] < (y + sigma_filter) else{
                continue
            }
            let d2 = pow((caf_f[i][1] - x), 2) + pow((caf_f[i][2] - y), 2)
            let score = exp(-0.5 * d2 / sigma2) * caf_f[i][0]
            if score >= score_1{
                score_2_i = score_1_i
                score_2 = score_1
                score_1_i = i
                score_1 = score
            }else if score > score_2{
                score_2_i = i
                score_2 = score
            }
        }
        let entry_1 = [caf_f[score_1_i][3], caf_f[score_1_i][4], caf_f[score_1_i][5], caf_f[score_1_i][6]]
        if score_2 < 0.01 || score_2 < 0.5 * score_1{
            return [entry_1[0], entry_1[1], entry_1[3], score_1 * 0.5]
        }
        let entry_2 = [caf_f[score_2_i][3], caf_f[score_2_i][4], caf_f[score_2_i][5], caf_f[score_2_i][6]]
        let blend_d2 = pow((entry_1[0] - entry_2[0]), 2) + pow((entry_1[1] - entry_2[1]), 2)
        if blend_d2 > pow(entry_1[3], 2) / 4.0{
            return [entry_1[0], entry_1[1], entry_1[3], score_1 * 0.5]
        }
        return [(score_1 * entry_1[0] + score_2 * entry_2[0]) / (score_1 + score_2),
                (score_1 * entry_1[1] + score_2 * entry_2[1]) / (score_1 + score_2),
                (score_1 * entry_1[3] + score_2 * entry_2[3]) / (score_1 + score_2),
                0.5 * (score_1 + score_2)
                ]
    }
    
    func connectionValue(start_i:Int, end_i:Int)->[Float]{
//        let cafScored = self.caf_forward[start_i]!
        let caf_f = self.cafScored.caf_f[start_i]!
        if caf_f.count == 0 {
            return [0.0, 0.0, 0.0, 0.0]
        }
        let ann = self.anns[start_i]!
        let new_xysv = grow_connection_blend(caf_f: caf_f, x: ann.x, y: ann.y, scale: ann.s)
        if new_xysv[3] == 0.0{
            return [0.0, 0.0, 0.0, 0.0]
        }
        let keypoint_score = (new_xysv[3] * ann.v).squareRoot()
        if keypoint_score < 0.15{
            return [0.0, 0.0, 0.0, 0.0]
        }
        if keypoint_score < ann.v * 0.5{
            return [0.0, 0.0, 0.0, 0.0]
        }
        return [new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score]
    }
    
    func frontier_get()->Frontier?{
        while (!self.frontier.isEmpty){
            let entry = self.frontier.popLast()!
            if entry.new_xysv.count != 0{
                return entry
            }
            let start_i = entry.start_i
            let end_i = entry.end_i
            if self.anns[end_i] != nil{
                continue
            }
            let new_xysv = self.connectionValue(start_i: start_i, end_i: end_i)
            if new_xysv[3]==0.0{
                continue
            }
            self.frontier.insert(Frontier(score: -new_xysv[3], new_xysv: new_xysv, start_i: start_i, end_i: end_i), at: 0)
        }
        return nil
    }
    
    func add_to_frontier(start_i: Int){
        for (end_i, vals) in self.by_source[start_i]! {
            if self.anns[end_i] != nil {
                continue
            }
            if self.in_frontier.contains(where: { $0 == (start_i, end_i) }) {
                continue
            }
            let max_possible_score = self.anns[start_i]!.v.squareRoot()
            self.frontier.insert(Frontier(score: -max_possible_score, new_xysv: [Float](), start_i: start_i, end_i: end_i), at: 0)
            self.in_frontier.insert((start_i, end_i), at: 0)
        }
    }

    func _grow(vfxys:[Float]){
        for start_i in self.anns.keys{
            add_to_frontier(start_i: start_i)
        }
        while(true){
            let entry = self.frontier_get()
            if entry == nil{
                break
            }
            let new_xysv = entry!.new_xysv
            let jsi = entry!.start_i
            let jti = entry!.end_i
            if self.anns[jti] != nil{
                continue
            }
            self.anns[jti] = Ann(v: new_xysv[3], x: new_xysv[0], y: new_xysv[1], s: new_xysv[2])
            add_to_frontier(start_i: jti)
        }
    }
    
    func cif_fill(cif_head: MLMultiArray){
        
        let n_field = cif_head.shape[1].intValue
        let ii = cif_head.shape[3].intValue
        let jj = cif_head.shape[4].intValue
        for i_field in 0..<n_field{
            for i in 0..<ii{
                for j in 0..<jj{
                    let multiArrayIndex = [0, i_field as NSNumber, 0, i as NSNumber, j as NSNumber]
                    guard cif_head[multiArrayIndex].floatValue > 0.5 else{
                        continue
                    }
//                                print(cif_head[multiArrayIndex])
                    let vIndex = [0, i_field as NSNumber, 0, i as NSNumber, j as NSNumber]
                    let xIndex = [0, i_field as NSNumber, 1, i as NSNumber, j as NSNumber]
                    let yIndex = [0, i_field as NSNumber, 2, i as NSNumber, j as NSNumber]
                    let sIndex = [0, i_field as NSNumber, 4, i as NSNumber, j as NSNumber]
                    let v = cif_head[vIndex].floatValue
                    let x = cif_head[xIndex].floatValue * 8.0
                    let y = cif_head[yIndex].floatValue * 8.0
                    let s = cif_head[sIndex].floatValue * 8.0
                    self.seeds.append([v, Float(i_field), x, y, s])
                }
            }
        }
        self.seeds = self.seeds.sorted { ($0[0] as Float) > ($1[0] as Float) }

    }
    
    func caf_fill(caf_head: MLMultiArray){
        let n_field = caf_head.shape[1].intValue
        let ii = caf_head.shape[3].intValue
        let jj = caf_head.shape[4].intValue

        for i_field in 0..<n_field{
            for i in 0..<ii{
                for j in 0..<jj{
                    let cIndex = [0, i_field as NSNumber, 0, i as NSNumber, j as NSNumber]
                    let c = caf_head[cIndex].floatValue
                    guard c > 0.2 else{
                        continue
                    }
                    print("passed \(i_field) " )
                    
                    let x1Index = [0, i_field as NSNumber, 1, i as NSNumber, j as NSNumber]
                    let y1Index = [0, i_field as NSNumber, 2, i as NSNumber, j as NSNumber]
                    let x2Index = [0, i_field as NSNumber, 3, i as NSNumber, j as NSNumber]
                    let y2Index = [0, i_field as NSNumber, 4, i as NSNumber, j as NSNumber]
                    let bIndex = [0, i_field as NSNumber, 6, i as NSNumber, j as NSNumber]
                    let scaleIndex = [0, i_field as NSNumber, 8, i as NSNumber, j as NSNumber]
                    let x1 = caf_head[x1Index].floatValue * 8.0
                    let y1 = caf_head[y1Index].floatValue * 8.0
                    let x2 = caf_head[x2Index].floatValue * 8.0
                    let y2 = caf_head[y2Index].floatValue * 8.0
                    let b = caf_head[bIndex].floatValue * 8.0
                    let scale = caf_head[scaleIndex].floatValue * 8.0
                    self.cafScored.caf_f[i_field]!.append([c, x1, y1, x2, y2, b, scale])
//                    self.caf_forward[i_field] = CafScored(caf: [c, x1, y1, x2, y2, b, scale])
//                    self.caf_forward[i_field]!.append([c, x1, y1, x2, y2, b, scale])
                }
            }
        }
    }
    
}


