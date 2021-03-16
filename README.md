# Image-stabilization-and-blur-classification
Input image stabilization and blur classification using OpenCV

### Image Stabilization:
Image Stabilization using optical Flow Pyramid between feature points calculated using OpenCV. 
Finally, smoothing out the transform trajectory and applying affine warping using the translation and rotation gradients. 

### Image Blur Classification
Blur Image classification by thresholding the variance of the laplacian for the image.
The reason this method works is due to the definition of the Laplacian operator itself, which is used to measure the 2nd derivative of an image. The Laplacian highlights regions of an image containing rapid intensity changes, much like the Sobel and Scharr operators. And, just like these operators, the Laplacian is often used for edge detection. 

The assumption here is that if an image contains high variance then there is a wide spread of responses, both edge-like and non-edge like, representative of a normal, in-focus image. But if there is very low variance, then there is a tiny spread of responses, indicating there are very little edges in the image. As we know, the more an image is blurred, the less edges there are.
But obviously this method is domain dependent and choosing the right threshold based on experiments is crucial.
