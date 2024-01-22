function Xz = myczt(xn, Mz, W, A)
%MYCZT perform chirp-z transform in the first dimension of xn
%
% Argument-----------------------------------------------------------------
% xn: input signal (1D, 2D, or 3D)
% Mz: length of output signal
% W : W = exp(-1i*2*pi*delta_fx/fs)
%     arbitrary step size for transform
% A : A = exp( 1i*2*pi*fx1/fs)
%     arbitrary starting point for transform
%
% Output-------------------------------------------------------------------
% Xz: output signal
%
% Xin Liu
% liuxin2018@zju.edu.cn
% Dec.19, 2020

%% check input
sz = size(xn);
oldm = sz(1);

if length(sz) > 3
    error('Invalid input dimensions!');
end

if length(sz) == 3 && sz(1) == 1 && sz(2) == 1  % thin array
    % Input is 1x1xK
    xn = reshape(xn,numel(xn),1);
    sz = size(xn);
    thinArray = true;
else
    thinArray = false;
end

if sz(1) == 1
    xn = permute(xn,[2,1,3:length(sz)]);
    sz = size(xn);
end

if any([size(Mz) size(W) size(A)]~=1)
    error('Invalid input dimensions!');
end

%% chirp z transform
% size of input signal
[Nx1, Nx2, Nx3] = size(xn);

kk = ((-Nx1+1):max(Mz-1,Nx1-1)).';  % full length for FFT (Nx+Mz-1)
ind_x = (0:(Nx1-1)).';  % index of the sequence of the first dimension of xn
WW = W.^((kk.^2)./2);
AW = A.^(-ind_x).*WW(Nx1+ind_x);  % chose the indices 'm+nn' of ww
AW = repmat(AW, [1,Nx2,Nx3]);

% two functions of n
fn = xn.*AW;
hn = WW(1:Nx1+Mz-1).^-1;

%% fast convolution via FFT
% length for power-of-two FFT
nfft = 2^nextpow2(Nx1+Mz-1);
Fr = fft(fn,nfft);
Hr = fft(hn,nfft);
Hr = repmat(Hr, [1,Nx2,Nx3]);

Gr = Fr.*Hr;  % multiplication in frequency domain
Xz = ifft(Gr);  % inverse fourier transform
Xz = Xz(Nx1:(Nx1+Mz-1),:,:);  % extract effective data

%% multiply prefix
prefix = repmat(WW(Nx1:Nx1+Mz-1), [1,Nx2,Nx3]);
Xz = prefix.*Xz;

%% reshape output
if oldm == 1
    Xz = permute(Xz,[2,1,3:length(sz)]);
end

if thinArray
    Xz = reshape(Xz,[1,1,numel(Xz)]);
end
end