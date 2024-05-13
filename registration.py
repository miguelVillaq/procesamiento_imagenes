import SimpleITK as sitk
import numpy as np
import nibabel as nib

def registration(fixed_img, mov_img, step, tol, repets):
    fixed_img = sitk.ReadImage(fixed_img)
    mov_img = sitk.ReadImage(mov_img)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(step, tol, repets) # 4.0, 0.01, 200
    
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_img.GetDimension()))
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    outTx = registration_method.Execute(fixed_img,mov_img)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    
    out = resampler.Execute(mov_img)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_img), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    
    mov = sitk.GetArrayFromImage(out)
    
    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {registration_method.GetOptimizerIteration()}")
    print(f" Metric value: {registration_method.GetMetricValue()}")
    
    return mov,out