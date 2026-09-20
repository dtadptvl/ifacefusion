import UIKit
import CoreImage

enum WarpTemplate {
    case arcface112v2,arcface128,dflWholeFace,ffhq512
    var normalized:[CGPoint] {
        switch self {
        case .arcface112v2:[.init(x:.34191607,y:.46157411),.init(x:.65653393,y:.45983393),.init(x:.500225,y:.64050536),.init(x:.37097589,y:.82469196),.init(x:.63151696,y:.82325089)]
        case .arcface128:[.init(x:.36167656,y:.40387734),.init(x:.63696719,y:.40235469),.init(x:.50019687,y:.56044219),.init(x:.38710391,y:.72160547),.init(x:.61507734,y:.72034453)]
        case .dflWholeFace:[.init(x:.35342266,y:.39285716),.init(x:.62797622,y:.39285716),.init(x:.48660713,y:.54017860),.init(x:.38839287,y:.68750011),.init(x:.59821427,y:.68750011)]
        case .ffhq512:[.init(x:.37691676,y:.46864664),.init(x:.62285697,y:.46912813),.init(x:.50123859,y:.61331904),.init(x:.39308822,y:.725411),.init(x:.61150205,y:.72490465)]
        }
    }
}
struct AffineWarp { let transform:CGAffineTransform; let inverse:CGAffineTransform }
enum ImageWarp {
    static let context=CIContext(options:[.cacheIntermediates:false])
    static func estimate(source:[CGPoint],template:WarpTemplate,size:CGSize)->AffineWarp {
        let target=template.normalized.map{CGPoint(x:$0.x*size.width,y:$0.y*size.height)}
        let s0=source[0],s1=source[1],t0=target[0],t1=target[1]
        let sv=CGPoint(x:s1.x-s0.x,y:s1.y-s0.y),tv=CGPoint(x:t1.x-t0.x,y:t1.y-t0.y)
        let scale=hypot(tv.x,tv.y)/max(hypot(sv.x,sv.y),.001)
        let angle=atan2(tv.y,tv.x)-atan2(sv.y,sv.x)
        var tr=CGAffineTransform(scaleX:scale,y:scale).rotated(by:angle)
        let mapped=source.map{$0.applying(tr)}
        let dx=zip(mapped,target).map{$1.x-$0.x}.reduce(0,+)/CGFloat(source.count)
        let dy=zip(mapped,target).map{$1.y-$0.y}.reduce(0,+)/CGFloat(source.count)
        tr=tr.concatenating(.init(translationX:dx,y:dy))
        return .init(transform:tr,inverse:tr.inverted())
    }
    static func crop(_ image:UIImage,geometry:FaceGeometry,template:WarpTemplate,size:CGSize)throws->(UIImage,AffineWarp) {
        guard let cg=image.normalizedCGImage else { throw FaceGeometryError.noFace }
        let warp=estimate(source:geometry.landmarks,template:template,size:size)
        let ci=CIImage(cgImage:cg).transformed(by:warp.transform),rect=CGRect(origin:.zero,size:size)
        guard let out=context.createCGImage(ci,from:rect) else { throw FaceGeometryError.noFace }
        return (UIImage(cgImage:out),warp)
    }
    static func paste(_ crop:UIImage,onto base:UIImage,warp:AffineWarp)throws->UIImage {
        guard let b=base.normalizedCGImage,let c=crop.normalizedCGImage else{return base}
        let baseCI=CIImage(cgImage:b),cropCI=CIImage(cgImage:c).transformed(by:warp.inverse),e=cropCI.extent
        let radius=min(e.width,e.height)*0.08
        guard let mask=CIFilter(name:"CIRoundedRectangleGenerator",parameters:["inputExtent":CIVector(cgRect:e.insetBy(dx:radius*.2,dy:radius*.2)),"inputRadius":radius])?.outputImage?.cropped(to:e) else{return base}
        let blend=CIFilter(name:"CIBlendWithMask",parameters:[kCIInputImageKey:cropCI,kCIInputBackgroundImageKey:baseCI,kCIInputMaskImageKey:mask])?.outputImage ?? baseCI
        guard let out=context.createCGImage(blend,from:baseCI.extent) else{return base}; return UIImage(cgImage:out)
    }
}