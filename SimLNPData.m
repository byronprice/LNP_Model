% SimLNPData.m

M = 5000;trainM = 4000;
D = 100;dim = sqrt(D);
R = 2;

gaborIm = gabor(5,45);
gaborIm = real(gaborIm.SpatialKernel);
gaborIm = gaborIm(10:50-1,35:75-1);

K = imresize(gaborIm,1/4)*7;

gaborIm = gabor(10,150);
gaborIm = real(gaborIm.SpatialKernel);
gaborIm = gaborIm(60:100-1,60:100-1);

K2 = imresize(gaborIm,1/4)*2.5;

K = [K(:),K2(:)];

Log1Exp = @(x) log(1+exp(x));
BasisFun = @(x,mu,stdev) exp(-(x-mu).*(x-mu)./(2*stdev));

basisSet{1} = [-2.5;0;2.5];
basisSet{2} = [4;4;4];

nBases = size(basisSet{1},1);

alpha = normrnd(0.5,2,[R*nBases,1]);

Y = zeros(M,1);
X = zeros(M,D);
for mm=1:M
    nGabors = 100;
    finalImage = zeros(dim*4,dim*4);
    for nn=1:nGabors
        wavelen = 4.5+rand*2;
        gaborIm = gabor(wavelen,0+rand*180);
        gaborIm = real(gaborIm.SpatialKernel);
        ind1 = round(15+rand*15);
        ind2 = round(15+rand*15);
        gaborIm = gaborIm(ind1:ind1+40-1,ind2:ind2+40-1);
        
        finalImage = finalImage+normrnd(0,1)*gaborIm;
    end
    finalImage = imresize(finalImage,1/4)+normrnd(0,1,[10,10]);
    
    reduce = K'*finalImage(:);
    Phi = zeros(R*nBases,1);

    index = 1;
    for ii=1:R
        Phi(index:index+nBases-1) = BasisFun(repmat(reduce(ii),[nBases,1]),basisSet{1},basisSet{2});
        index = index+nBases;
    end
    Y(mm) = Log1Exp(Phi'*alpha);
    X(mm,:) = finalImage(:);
end

Y = poissrnd(Y); % add poisson spiking variability

logLike = -Inf;

for ii=1:10
    [kest,alphaest,loglike] = FitLNP(Y(1:trainM),X(1:trainM,:),2); % estimate parameters of the model
    
    if loglike>logLike
        logLike = loglike;
        Kest = kest;
        alphaEst = alphaest;
    end
end

% get model fit on held-out test data

getDev = @(y,mu) y.*log(y./mu)-(y-mu);

mu = mean(Y(trainM+1:end));
nullDeviance = getDev(Y(trainM+1:end),mu);
tmp = -(Y(trainM+1:end)-mu);
nullDeviance(isnan(nullDeviance)) = tmp(isnan(nullDeviance));
nullDeviance = 2*sum(nullDeviance);

mu = zeros(length(Y(trainM+1:end)),1);

count = 1;
for mm=trainM+1:M
    reduce = Kest'*X(mm,:)';
    Phi = zeros(R*nBases,1);

    index = 1;
    for ii=1:R
        Phi(index:index+nBases-1) = BasisFun(repmat(reduce(ii),[nBases,1]),basisSet{1},basisSet{2});
        index = index+nBases;
    end
    mu(count) = Log1Exp(Phi'*alphaEst);
   
    count = count+1;
end

modelDeviance = getDev(Y(trainM+1:end),mu);
tmp = -(Y(trainM+1:end)-mu);
modelDeviance(isnan(modelDeviance)) = tmp(isnan(modelDeviance));
modelDeviance = 2*sum(modelDeviance);

heldOutExpDev = 1-modelDeviance/nullDeviance