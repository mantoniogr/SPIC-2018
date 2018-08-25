% Image qualificators

% ssim
% Structural Similarity Index (SSIM) for measuring image quality

% ssimval = ssim(A,ref)

% A = imread('filtered/venus-6/disp6-venus-noise-MCbR.png');
% A = imread('filtered/venus-6/Filtered_average6-venus4.png');
% A = imread('filtered/venus-6/Anisotropic Diffusion.png');
% A = imread('filtered/venus-6/Bilateral Filter.png');
% A = imread('filtered/venus-6/Bilateral Upsampling.png');
% A = imread('filtered/venus-6/Layered Bilateral Filter.png');
% A = imread('filtered/venus-6/Markov Random Field(Kernel Data Term).png');
% A = imread('filtered/venus-6/Markov Random Field(Tensor).png');
% A = imread('filtered/venus-6/Noise-aware Filter.png');
A = imread('filtered/venus-6/Weight Mode Filter.png');

ref = imread('filtered/venus-6/disp6-venus.pgm');

ssimval = ssim(A(:,:,1),ref(:,:,1))
% ssimval = ssim(A, ref)

% psnr
% Peak Signal-to-Noise Ratio (PSNR)
% peaksnr = psnr(A,ref)
peaksnr = psnr(A(:,:,1),ref(:,:,1))

% mse
% Mean-squared error
% err = immse(A, ref)
err = immse(A(:,:,1),ref(:,:,1))