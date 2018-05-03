function Test_FMapNLSum(cid)

if nargin < 1
  cid = 'double';
end

vec=@(x)x(:);range=@(x)[min(vec(x)),max(vec(x))];

H=200;W=100;D=8;N=5;Nb=4;
%H=6;W=1;D=2;N=1;Nb=3;
X=randn(H,W,D,N,cid); idx=randsrc(H*W*Nb*N,1,1:H*W); idx = reshape(idx,H,W,Nb,N);
idx(:,:,1,1) = reshape(1:H*W,H,W); 

for n = 1:N
    idx(:,:,1,n)=idx(:,:,1,1); 
end
idx = uint32(idx);


[widx,n_table,I]=misc.FMapNLSumT_helper(idx);

fprintf('------------------------------------------------------\n');
fprintf('\t\t\tCase 1\n');
fprintf('------------------------------------------------------\n');

fprintf('*---------Forward Operation-----------*\n');

Y = zeros(size(X),'like',X);
Weights = randn(H,W,Nb,N,'like',X);
start_time = tic;
for i = 1:H 
  for j=1:W
    for d=1:D 
      for n=1:N
        for r=1:Nb 
          Y(i,j,d,n) = Y(i,j,d,n) + Weights(i,j,r,n)*X(idx(i,j,r,n)+(d-1)*H*W+(n-1)*H*W*D);
        end
      end
    end
  end
end
end_time = toc(start_time);

fprintf('Execution time for for-loop : %d secs\n',end_time);
start_time = tic;
Y2 = FMapNLSum_cpu(X,Weights,idx-1);
end_time = toc(start_time); % We substract 1 from idx since
% in C++ the starting element is zero and not 1.
fprintf('Execution time for mex-cpu : %d secs\n',end_time);
fprintf('for-loop vs mex-cpu implementation minimum and maximum difference : [%d %d]\n',range(Y-Y2));

start_time = tic;
Y2g = FMapNLSum_gpu(gpuArray(X),gpuArray(Weights),gpuArray(idx-1));
end_time = toc(start_time); % We substract 1 from idx since
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Y2-gather(Y2g)));

fprintf('\n*---------Transpose Operation-----------*\n');

start_time = tic;
Z = FMapNLSumT_cpu(Y2,Weights,widx-1,n_table,I-1);
end_time = toc(start_time);
fprintf('Execution time for mex-cpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-cpu : %d\n',Z(:)'* X(:) - Y2(:)'*Y2(:));
start_time = tic;
Zg = FMapNLSumT_gpu(Y2g,gpuArray(Weights),gpuArray(widx-1),gpuArray(n_table),gpuArray(I-1));
end_time = toc(start_time);
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-gpu : %d\n',Zg(:)'* gpuArray(X(:)) - Y2g(:)'*Y2g(:));

fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Z-gather(Zg)));

fprintf('\n\n------------------------------------------------------\n');
fprintf('\t\t\tCase 2\n');
fprintf('------------------------------------------------------\n');

fprintf('*---------Forward Operation-----------*\n');

Y = zeros(size(X),'like',X);
Weights = randn(Nb,1,'like',X);
start_time = tic;
for i = 1:H 
  for j=1:W
    for d=1:D 
      for n=1:N
        for r=1:Nb 
          Y(i,j,d,n) = Y(i,j,d,n) + Weights(r)*X(idx(i,j,r,n)+(d-1)*H*W+(n-1)*H*W*D);
        end
      end
    end
  end
end
end_time = toc(start_time);

fprintf('Execution time for for-loop : %d secs\n',end_time);
start_time = tic;
Y2 = FMapNLSum_cpu(X,Weights,idx-1);
end_time = toc(start_time); % We substract 1 from idx since
% in C++ the starting element is zero and not 1.
fprintf('Execution time for mex-cpu : %d secs\n',end_time);
fprintf('for-loop vs mex-cpu implementation minimum and maximum difference : [%d %d]\n',range(Y-Y2));

start_time = tic;
Y2g = FMapNLSum_gpu(gpuArray(X),gpuArray(Weights),gpuArray(idx-1));
end_time = toc(start_time); % We substract 1 from idx since
fprintf('Execution time for mex-cgu : %d secs \n',end_time);
fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Y2-gather(Y2g)));


fprintf('\n*---------Transpose Operation-----------*\n');

start_time = tic;
Z = FMapNLSumT_cpu(Y2,Weights,widx-1,n_table,I-1);
end_time = toc(start_time);
fprintf('Execution time for mex-cpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-cpu : %d\n',Z(:)'* X(:) - Y2(:)'*Y2(:));
start_time = tic;
Zg = FMapNLSumT_gpu(Y2g,gpuArray(Weights),gpuArray(widx-1),gpuArray(n_table),gpuArray(I-1));
end_time = toc(start_time);
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-gpu : %d\n',Zg(:)'* gpuArray(X(:)) - Y2g(:)'*Y2g(:));

fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Z-gather(Zg)));

fprintf('\n\n------------------------------------------------------\n');
fprintf('\t\t\tCase 3\n');
fprintf('------------------------------------------------------\n');

fprintf('*---------Forward Operation-----------*\n');

Y = zeros(size(X),'like',X);
Weights = randn(D*Nb,1,'like',X);
start_time = tic;
for i = 1:H 
  for j=1:W
    for d=1:D 
      for n=1:N
        for r=1:Nb 
          Y(i,j,d,n) = Y(i,j,d,n) + Weights((r-1)*D+d)*X(idx(i,j,r,n)+(d-1)*H*W+(n-1)*H*W*D);
        end
      end
    end
  end
end
end_time = toc(start_time);

fprintf('Execution time for for-loop : %d secs\n',end_time);
start_time = tic;
Y2 = FMapNLSum_cpu(X,Weights,idx-1);
end_time = toc(start_time); % We substract 1 from idx since
% in C++ the starting element is zero and not 1.
fprintf('Execution time for mex-cpu : %d secs\n',end_time);
fprintf('for-loop vs mex-cpu implementation minimum and maximum difference : [%d %d]\n',range(Y-Y2));

start_time = tic;
Y2g = FMapNLSum_gpu(gpuArray(X),gpuArray(Weights),gpuArray(idx-1));
end_time = toc(start_time); % We substract 1 from idx since
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Y2-gather(Y2g)));

fprintf('\n*---------Transpose Operation-----------*\n');

start_time = tic;
Z = FMapNLSumT_cpu(Y2,Weights,widx-1,n_table,I-1);
end_time = toc(start_time);
fprintf('Execution time for mex-cpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-cpu : %d\n',Z(:)'* X(:) - Y2(:)'*Y2(:));
start_time = tic;
Zg = FMapNLSumT_gpu(Y2g,gpuArray(Weights),gpuArray(widx-1),gpuArray(n_table),gpuArray(I-1));
end_time = toc(start_time);
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-gpu : %d\n',Zg(:)'* gpuArray(X(:)) - Y2g(:)'*Y2g(:));

fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Z-gather(Zg)));

fprintf('\n\n------------------------------------------------------\n');
fprintf('\t\t\tCase 4\n');
fprintf('------------------------------------------------------\n');

fprintf('*---------Forward Operation-----------*\n');

Y = zeros(size(X),'like',X);
Weights = randn(H,W,D*Nb,N,'like',X);
start_time = tic;
for i = 1:H 
  for j=1:W
    for d=1:D 
      for n=1:N
        for r=1:Nb 
          Y(i,j,d,n) = Y(i,j,d,n) + Weights(i,j,(r-1)*D+d,n)*X(idx(i,j,r,n)+(d-1)*H*W+(n-1)*H*W*D);
        end
      end
    end
  end
end
end_time = toc(start_time);

fprintf('Execution time for for-loop : %d secs\n',end_time);
start_time = tic;
Y2 = FMapNLSum_cpu(X,Weights,idx-1);
end_time = toc(start_time); % We substract 1 from idx since
% in C++ the starting element is zero and not 1.
fprintf('Execution time for mex-cpu : %d secs\n',end_time);
fprintf('for-loop vs mex-cpu implementation minimum and maximum difference : [%d %d]\n',range(Y-Y2));

start_time = tic;
Y2g = FMapNLSum_gpu(gpuArray(X),gpuArray(Weights),gpuArray(idx-1));
end_time = toc(start_time); % We substract 1 from idx since
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Y2-gather(Y2g)));

fprintf('\n*---------Transpose Operation-----------*\n');

start_time = tic;
Z = FMapNLSumT_cpu(Y2,Weights,widx-1,n_table,I-1);
end_time = toc(start_time);
fprintf('Execution time for mex-cpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-cpu : %d\n',Z(:)'* X(:) - Y2(:)'*Y2(:));
start_time = tic;
Zg = FMapNLSumT_gpu(Y2g,gpuArray(Weights),gpuArray(widx-1),gpuArray(n_table),gpuArray(I-1));
end_time = toc(start_time);
fprintf('Execution time for mex-gpu : %d secs \n',end_time);
fprintf('Inner-product difference for mex-gpu : %d\n',Zg(:)'* gpuArray(X(:)) - Y2g(:)'*Y2g(:));

fprintf('cpu vs gpu mex implementation minimum and maximum difference : [%d %d]\n',range(Z-gather(Zg)));