function Eout = FresnelDiffraction_CZT(Beam,Scope,Option)
%FRESNELDIFFRACTION_CZT calculate Fresnel Diffraction by chirp-z transform
%(CZT) in Cartesian coordinate system.
%
% INPUT********************************************************************
% Beam.wavelength: scalar value, wavelength of light
% Beam.amp: M*N matrix, amplitude distribution on diffraction aperture
% Beam.phs: M*N matrix, phase distribution on diffraction aperture
% Beam.pixelsize: scalar value, pixel size of diffraction aperture after
% discretization
% Scope.xs: 1*N array, representing x axis of observation plane
% Scope.ys: 1*N array, representing y axis of observation plane
% Scope.zs: 1*N array, representing z axis of observation volume
% OptionStruct.UseGpu: 0 or 1, option using GPU acceleration
% OptionStruct.Precision: 0 or 1, precision of numbers
%
% OUTPUT*******************************************************************
% Eout: diffraction field on observation plane
%
% REFERENCES***************************************************************
% [1] "Hu Y, Wang Z, Wang X, et al. Efficient full-path optical calculation
% of scalar and vector diffraction using the Bluestein method[J]. Light:
% Science & Applications, 2020, 9(1): 119."
%
% *************************************************************************
% LIU Xin
% liuxin2018@zju.edu.cn
% Apr.23, 2021

%% data initialization
n = Option.n;

if strcmp(Option.Precision,'single')
    n = single(n);
    Beam.wavelength = single(Beam.wavelength);
    Beam.amp = single(Beam.amp);
    Beam.phs = single(Beam.phs);
    Beam.pixelsize = single(Beam.pixelsize);
    Scope.xs = single(Scope.xs);
    Scope.ys = single(Scope.ys);
    Scope.zs = single(Scope.zs);
end

if Option.UseGpu == 1
    n = gpuArray(n);
    Beam.wavelength = gpuArray(Beam.wavelength);
    Beam.amp = gpuArray(Beam.amp);
    Beam.phs = gpuArray(Beam.phs);
    Beam.pixelsize = gpuArray(Beam.pixelsize);
    Scope.xs = gpuArray(Scope.xs);
    Scope.ys = gpuArray(Scope.ys);
    Scope.zs = gpuArray(Scope.zs);
end

lambda = Beam.wavelength/n;  % wavelength in medium

%% Diffraction aperture
% resolution of diffraction aperture
[LRy, LRx] = size(Beam.amp);

E0 = Beam.amp.*exp(1i*Beam.phs);

% real size of diffraction aperture
LSx = (LRx-1)*Beam.pixelsize;
LSy = (LRy-1)*Beam.pixelsize;

% real coordinates of diffraction aperture
xd = linspace(-LSx/2,LSx/2,LRx);
yd = linspace(-LSy/2,LSy/2,LRy);
[xxd,yyd] = meshgrid(xd,yd);

%% Observation plane
% real coordinates of observation plane
[xxs,yys] = meshgrid(Scope.xs,Scope.ys);

% size of observation plane
lx = length(Scope.xs); ly = length(Scope.ys); lz = length(Scope.zs);

if lx ~= 1
    pixelSizeX = Scope.xs(2)-Scope.xs(1);
else
    pixelSizeX = 0;
end

if ly ~= 1
    pixelSizeY = Scope.ys(2)-Scope.ys(1);
else
    pixelSizeY = 0;
end

k = 2*pi/lambda;

%% Fresnel Diffraction Calculation
Eout = zeros(ly,lx,lz);
if strcmp(Option.Precision,'single')
    Eout = single(Eout);
end

if Option.UseGpu == 1
    Eout = gpuArray(Eout);
end
zs = reshape(Scope.zs,1,1,lz);

% Equation 4
F0 = exp(1i*k*zs)./(1i*Beam.wavelength*zs/n).*...
    exp(1i*k*(xxs.^2+yys.^2)./(2*zs));

% sampling frequency in 'k-space' (overall size of observation plane)
fs = (Beam.wavelength/n)*zs/Beam.pixelsize;

fx1 = Scope.xs(1);  % start point in x direction
fy1 = Scope.ys(1);  % start point in y direction
Ax = exp( 1i*2*pi*fx1./fs);  % Equation S15
Ay = exp( 1i*2*pi*fy1./fs);  % Equation S15
Wx = exp(-1i*2*pi*pixelSizeX./fs);  % Equation S16
Wy = exp(-1i*2*pi*pixelSizeY./fs);  % Equation S16

Mshiftx = -(LRx-1)/2;  % Equation S18
Mshifty = -(LRy-1)/2;  % Equation S18

% Equation S19
[xss,yss] = meshgrid(Scope.xs.*Mshiftx,Scope.ys.*Mshifty);
Pshift = exp(-1i*2*pi*(xss+yss)./fs);

if Option.TimeIt == 1
    tic;
end

for ii = 1:lz
    % Equation 5
    F = exp(1i*k*(xxd.^2+yyd.^2)./(2*zs(ii)));
    
    % Equation 6
    E = E0.*F;
    
    % one-dimensional CZT in y direction
    EHold = myczt(E,ly,Wy(ii),Ay(ii));
    EHold = EHold.';
    
    % one-dimensional CZT in x direction
    EHold = myczt(EHold,lx,Wx(ii),Ax(ii));
    EHold = EHold.';
    
    EHold = EHold.*Pshift(:,:,ii);  % phase shift correction
    
    Eout(:,:,ii) = F0(:,:,ii).*EHold;  % Equation 6
end

if Option.TimeIt == 1
    toc;
end

if Option.UseGpu == 1
    Eout = gather(Eout);
end

if strcmp(Option.Precision,'single')
    Eout = double(Eout);
end
end