clear;
Findex=[1,3:30];
para.dim=100;
para.p_best_rate=0.11 ;
para.Xmin=-100;
para.Xmax=100;
para.pop_size=round(para.dim*18);
para.arc_rate= 2.6; 
para.memo_size= 6;
para.maxfe=10000*para.dim;
para.OPTIMUM =0;
para.EPSILON=10^(-8);
repeat=1;
% for zht =1:18
% num=Findex(zht);
hidden_num=100;
norm_w=1;
load('.\net_10D')
tic
for iter = 1:29
    problem_index=Findex(iter);

        for i=1:11
        [ bsf_fitness(i),processvalue(:,i) ]=RL_LSHADE( problem_index,repeat,para ,net_w,norm_w,hidden_num);
        end
%     location=['D:\DE\ADALSHADE\data\10D\net\RL_F_',num2str(problem_index)];
%     save(location,'bsf_fitness')
end
toc