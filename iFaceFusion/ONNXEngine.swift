import Foundation
import UIKit
import OnnxRuntimeBindings

actor ONNXEngine {
    static let shared=ONNXEngine()
    private let env:ORTEnv
    private var sessions:[String:ORTSession]=[:]
    init() { env=try! ORTEnv(loggingLevel:.warning) }

    private func session(path:String)throws->ORTSession {
        if let s=sessions[path] { return s }
        let options=try ORTSessionOptions()
        try? options.setIntraOpNumThreads(2)
        let coreML=ORTCoreMLExecutionProviderOptions(); coreML.enableOnSubgraphs=true
        try? options.appendCoreMLExecutionProvider(with:coreML)
        let s=try ORTSession(env:env,modelPath:path,sessionOptions:options); sessions[path]=s; return s
    }

    func runImage(modelURL:URL,image:UIImage,inputSize:CGSize,normalization:ClosedRange<Float>=0...1)throws->UIImage {
        let s=try session(path:modelURL.path)
        guard let inputName=try s.inputNames().first,let outputName=try s.outputNames().first else { throw InferenceError.io }
        let floats=try TensorImage.chw(image,size:inputSize,range:normalization)
        let data=NSMutableData(bytes:floats,length:floats.count*MemoryLayout<Float>.size)
        let value=try ORTValue(tensorData:data,elementType:.float,shape:[1,3,NSNumber(value:Float(inputSize.height)),NSNumber(value:Float(inputSize.width))])
        let outputs=try s.run(withInputs:[inputName:value],outputNames:Set([outputName]),runOptions:nil)
        guard let out=outputs[outputName] else { throw InferenceError.io }
        let info=try out.tensorTypeAndShapeInfo(),bytes=try out.tensorData()
        return try TensorImage.image(data:bytes,shape:info.shape)
    }
}
enum InferenceError:LocalizedError {
    case io,unsupported(String)
    var errorDescription:String? { switch self { case .io:"The model returned an unreadable tensor."; case .unsupported(let s):s } }
}
enum TensorImage {
    static func chw(_ image:UIImage,size:CGSize,range:ClosedRange<Float>)throws->[Float] {
        guard let cg=image.normalizedCGImage else{throw InferenceError.io}
        let w=Int(size.width),h=Int(size.height),cs=CGColorSpaceCreateDeviceRGB()
        var rgba=[UInt8](repeating:0,count:w*h*4)
        guard let ctx=CGContext(data:&rgba,width:w,height:h,bitsPerComponent:8,bytesPerRow:w*4,space:cs,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else{throw InferenceError.io}
        ctx.interpolationQuality=.high; ctx.draw(cg,in:CGRect(x:0,y:0,width:w,height:h))
        var out=[Float](repeating:0,count:w*h*3)
        for i in 0..<(w*h) { for c in 0..<3 { let v=Float(rgba[i*4+c])/255; out[c*w*h+i]=range.lowerBound+v*(range.upperBound-range.lowerBound) } }
        return out
    }
    static func image(data:NSMutableData,shape:[NSNumber])throws->UIImage {
        guard shape.count>=4 else{throw InferenceError.io}
        let h=shape[shape.count-2].intValue,w=shape[shape.count-1].intValue
        let count=data.length/MemoryLayout<Float>.size
        guard count>=3*w*h else{throw InferenceError.io}
        let p=data.bytes.bindMemory(to:Float.self,capacity:count)
        var rgba=[UInt8](repeating:255,count:w*h*4)
        for i in 0..<(w*h) { for c in 0..<3 { rgba[i*4+c]=UInt8(max(0,min(255,Int(p[c*w*h+i]*255)))) } }
        let provider=CGDataProvider(data:Data(rgba) as CFData)!
        let cg=CGImage(width:w,height:h,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:w*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:true,intent:.defaultIntent)!
        return UIImage(cgImage:cg)
    }
}