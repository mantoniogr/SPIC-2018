% Image qualificators

% ssim
% Structural Similarity Index (SSIM) for measuring image quality

% ssimval = ssim(A,ref)

A = imread('Weight Mode Filter.png');
ref = imread('teddy2_manual3.png');

ssimval = ssim(A,ref(:,:,1))
% ssimval = ssim(A, ref)

% psnr
% Peak Signal-to-Noise Ratio (PSNR)

% peaksnr = psnr(A,ref)

peaksnr = psnr(A,ref(:,:,1))

% mse
% Mean-squared error

% err = immse(A, ref)

err = immse(A,ref(:,:,1))