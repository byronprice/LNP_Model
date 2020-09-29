function [K,alpha,logLike] = FitLNP(Y,X,R)
% FitLNP.m
%   linear-nonlinear-poisson model from "Equivalence of
%    Information-Theoretic and Likelihood-Based Methods for Neural
%    Dimensionality Reduction" William, Sahani, Pillow 2015

%INPUTS: Y - neural data, size M-by-1 (trials)
%        X - stimulus (vectorized images other data for receptive field
%        mapping, size M-by-D (trials by dimensionality of stimulus)
%        R - rank of fit (# of linear filters to model)
%
%OUTPUTS: K - receptive fields from model fit, size D-by-R
%               (reshape after fit to visualize)
%         alpha - parameters that govern the shape of a non-linearity that
%           controls the shape of the response once the data has been
%           projected down using the linear mapping from the receptive field
%           (K'*X')
%
%Created: 2020/09/24
%  Byron Price
%Updated: 2020/09/24
% By: Byron Price

% initialize params
[M,D] = size(X);
K = normrnd(0,1,[D,R]);

basisSet{1} = repmat([-2.5;0;2.5],[1,M]);
basisSet{2} = repmat([4;4;4],[1,M]);

nBases = size(basisSet{1},1);
alpha = normrnd(0,1,[R*nBases,1]);

oneVec = ones(M,1);

Ksize = size(K);

x0 = [K(:);alpha];

fun = @(x) ObjectiveFun(x,Y,X,M,D,R,Ksize,oneVec,basisSet,nBases);

options = optimoptions('fminunc','Display','off','Algorithm','trust-region',...
    'MaxIterations',1e3,'SpecifyObjectiveGradient',true,'HessianFcn','objective'); % 'objective'

x = fminunc(fun,x0,options);
% x = fminunc(fun,x0,options);

K = x(1:numel(K));alpha = x(numel(K)+1:end);

K = reshape(K,Ksize);

logLike = -fun(x);
end

function [negloglike,gradient,hessian] = ObjectiveFun(x,Y,X,M,D,R,Ksize,oneVec,basisSet,nBases)
Klen = prod(Ksize);
K = x(1:Klen);alpha = x(Klen+1:end);np = length(x);

% nAlpha = np-Klen;
K = reshape(K,Ksize);

[lambda,Phi,PhiAlpha,Z] = GetLambda(X,K,alpha,basisSet,R,nBases,M);

loglikelihood = GetLikelihood(Y,lambda);

negloglike = -loglikelihood;

% get gradient
gradientComp = (Y./lambda)-oneVec;
gradientComp2 = -Y./(lambda.*lambda);

gradient = zeros(np,M);

index = 1;
lambdaPrime = Log1ExpDeriv1(PhiAlpha);
lambdaPrime2 = Log1ExpDeriv2(PhiAlpha);

PhiPrime = GetPhiPrime(Z,basisSet,R,nBases,M);
PhiPrime2 = GetPhiPrime2(Z,basisSet,R,nBases,M);

kAlpha = cell(R,1);
kk = cell(R,R);
for rr=1:R
    gradient(index:index+D-1,:) = bsxfun(@times,X',(lambdaPrime.*(PhiPrime{rr}*alpha))');
    
%     for aa=1:nAlpha
%         kAlpha{rr,aa} = Phi(:,aa)'.*(lambdaPrime2.*(PhiPrime{rr}*alpha(aa)))'+...
%             PhiPrime{rr}(:,aa)'.*lambdaPrime';
%     end
    
    kAlpha{rr} = bsxfun(@times,Phi',(lambdaPrime2.*(PhiPrime{rr}*alpha))')+...
        bsxfun(@times,PhiPrime{rr}',lambdaPrime');
        
    for ss=rr:R
        tmp = (lambdaPrime2.*(PhiPrime{rr}*alpha).*(PhiPrime{ss}*alpha))'+...
           (lambdaPrime.*(PhiPrime2{rr,ss}*alpha))';
        kk{rr,ss} = tmp;
    end
    index = index+D;
end

for jj=1:R
    for ii=jj+1:R
        kk{ii,jj} = kk{jj,ii};
    end
end

alphaGrad = bsxfun(@times,Phi',lambdaPrime');
gradient(index:end,:) = alphaGrad;

preGradient = gradient;
gradient = -preGradient*gradientComp;

% get hessian matrix
hessian = zeros(np,np);

alphaAlpha = bsxfun(@times,Phi',sqrt(lambdaPrime2'));

rInd = 1;
for ii=1:np
    sInd = rInd;
    for jj=ii:np
        hessian(ii,jj) = (preGradient(ii,:).*preGradient(jj,:))*gradientComp2;
        
        if ii<=Klen && jj<=Klen
             xInd = mod(ii-1,D)+1;
             yInd = mod(jj-1,D)+1;
             
             tmp = (X(:,xInd)'.*kk{rInd,sInd}.*X(:,yInd)');
             hessian(ii,jj) = hessian(ii,jj)+tmp*gradientComp;
        elseif ii<=Klen && jj>Klen
            xInd = mod(ii-1,D)+1;
            aInd = jj-Klen;
            hessian(ii,jj) = hessian(ii,jj)+(X(:,xInd)'.*kAlpha{rInd}(aInd,:))*gradientComp;
        elseif ii>Klen && jj>Klen
            aInd = ii-Klen;
            bInd = jj-Klen;
            hessian(ii,jj) = hessian(ii,jj)+(alphaAlpha(aInd,:).*alphaAlpha(bInd,:))*gradientComp;
        end
        
        if (mod(jj-1,D)+1)==D
            sInd = sInd+1;
        end
        
        if ii==jj
            hessian(ii,jj) = hessian(ii,jj)/2;
        end
    end
    
    if (mod(ii-1,D)+1)==D
        rInd = rInd+1;
    end
    
end

hessian = -(hessian+hessian');
end

function [Phi] = GetPhi(z,basisSet,nFilts,nBases,M)
Phi = zeros(M,nFilts*nBases);

index = 1;
for ii=1:nFilts
    Phi(:,index:index+nBases-1) = BasisFun(repmat(z(ii,:),[nBases,1]),basisSet)';
    index = index+nBases;
end

end

function [PhiPrime] = GetPhiPrime(z,basisSet,nFilts,nBases,M)

PhiPrime = cell(nFilts,1);

index = 1;
for ii=1:nFilts
    PhiPrime{ii} = zeros(M,nFilts*nBases);
    PhiPrime{ii}(:,index:index+nBases-1) = BasisDeriv(repmat(z(ii,:),[nBases,1]),basisSet)';
    index = index+nBases;
end

end

function [PhiPrime2] = GetPhiPrime2(z,basisSet,nFilts,nBases,M)

PhiPrime2 = cell(nFilts,nFilts);

for ii=1:nFilts
    for jj=1:nFilts
        PhiPrime2{ii,jj} = zeros(M,nFilts*nBases);
        if ii==jj
            index = 1;
            for kk=1:nFilts
                if kk==ii 
                    PhiPrime2{ii,jj}(:,index:index+nBases-1) = BasisDeriv2(repmat(z(kk,:),[nBases,1]),basisSet)';
                end
                index = index+nBases;
            end
        end
    end
end

end

function [basis] = BasisFun(z,basisSet)
basis = exp(-(z-basisSet{1}).*(z-basisSet{1})./(2*basisSet{2}));
end

function [basisprime] = BasisDeriv(z,basisSet)
basisprime = -BasisFun(z,basisSet).*(z-basisSet{1})./basisSet{2};
end

function [basisprime2] = BasisDeriv2(z,basisSet)
basisprime2 = -BasisDeriv(z,basisSet).*(z-basisSet{1})./basisSet{2}-...
    BasisFun(z,basisSet)./basisSet{2};
end

function [lambda,Phi,PhiAlpha,Z] = GetLambda(X,K,alpha,basisSet,nFilts,nBases,M)

Z = K'*X'; % nFilts -by- M (must expand to nFilts*nBases -by- M)

Phi = GetPhi(Z,basisSet,nFilts,nBases,M);
PhiAlpha = Phi*alpha;

lambda = Log1Exp(PhiAlpha); % M-by-1

end


function [loglikelihood] = GetLikelihood(Y,lambda)

loglikelihood = sum(Y.*log(lambda)-lambda);

end

function [y] = Log1Exp(x)
%   Calculate the softplus function of the input x
%    softplus(x) = log(1+exp(x))
if x<-34 % condition for small x
   y = 0;
else
   y = log(1+exp(-x))+x; % numerically stable calculation of log(1+exp(x))
end

end

function [yprime] = Log1ExpDeriv1(x)
yprime = 1./(1+exp(-x));
end

function [yprime2] = Log1ExpDeriv2(x)
yprime2 = Log1ExpDeriv1(x).*(1-Log1ExpDeriv1(x));
end