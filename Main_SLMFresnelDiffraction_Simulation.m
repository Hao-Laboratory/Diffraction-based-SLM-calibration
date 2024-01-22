clear;clc;
close all;

%% system parameters
Option.UseGpu = 0;
Option.Precision = 'double';
Option.TimeIt = 1;
Option.BatchSize = 32;
Option.n = 1;

ifFig = 1;

%% SLM parameters
s = 1;
SLM.Row = 1024*s;  % resolution of SLM
SLM.Col = 1272*s;

SLM.PixelSize = 12.5/s;  % pixel size of SLM 'um' (12.5um)
[xx,yy] = meshgrid(-(SLM.Col-1)/2:(SLM.Col-1)/2,...
    -(SLM.Row-1)/2:(SLM.Row-1)/2);

xr = xx.*SLM.PixelSize;
yr = yy.*SLM.PixelSize;

R0 = max(xr(:));
[phi,rho] = cart2pol(xr,yr);

% amplitude
% beta = 0.3*1e-3;
beta = 1e-4;

x0 = 0; y0 = 0;  % no offset
% x0 = 0; y0 = R0/2;  % vertical offset
% x0 = R0/2; y0 = 0;  % horizontal offset
% x0 = R0/2; y0 = R0/2;  % pi/4 offset
sigma_x = 1/beta;
sigma_y = 1/beta;
FWHM = 2.355*sigma_x

% amplitude
% Beam.amp = exp(-(beta*rho).^2);  % simple Gauss distribution
Beam.amp = 1/(2*pi*sigma_x*sigma_y)*exp(-0.5*(((xr-x0)/sigma_x).^2 + ((yr-y0)/sigma_y).^2));  % complicated Gauss distribution

Beam.wavelength = 785e-3;  % 785nm
Beam.pixelsize = SLM.PixelSize;

%% camera parameters
d = 300e3;

s1 = 1;
s2 = 3.5;
% s2 = 1e3;  % for far-field diffraction
pixelSize = 3.45/s1*s2;  % pixel size of camera (3.45um)

% resolution of camera
cx = 1440*s1;
cy = 1080*s1;

Lx = (cx-1)*pixelSize;
Ly = (cy-1)*pixelSize;

Scope.xs = -Lx/2:pixelSize:Lx/2;
Scope.ys = -Ly/2:pixelSize:Ly/2;
Scope.zs = d;

%% figure flag
if ifFig == 1
    figure;
    tl_result = tiledlayout(2,2,'TileSpacing','compact');
end

%% zero order
N = 101;
PhaseValues = linspace(0,2*pi,N);
% PhaseValues = pi;

phs = zeros(SLM.Row,SLM.Col);

for ii = PhaseValues

            phs(xx>0) = ii;

    Beam.phs = phs;
    
    Eout = FresnelDiffraction_CZT(Beam,Scope,Option);
    
    img = abs(Eout).^2;
    
    if ifFig == 1
        title(tl_result,['Phase ',num2str(ii,'%.2f')]);
        
        nexttile(tl_result,1);
        imagesc(Beam.amp);
        axis image off;
        colorbar;
        title('beam amplitude');
        
        nexttile(tl_result,2);
        imagesc(mod(Beam.phs,2*pi));
        axis image off;
        colorbar;
        title('SLM phase');
        
        nexttile(tl_result,3);
        imagesc(Scope.xs,Scope.ys,img);
        axis image off;
        colormap(gca,'gray');
        colorbar;
        title('camera intensity');
        
        nexttile(tl_result,4);
        plot(sum(img,1));
        axis tight;
        grid on;
        title('line profile');
        
        drawnow;
    end
    
    fprintf('phase %.4f\n',ii);
end