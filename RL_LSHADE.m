function [ bsf_fitness ,processvalue,po] = RL_LSHADE( problem_index,repeat,para,net_w,norm_w,hidden_num )

    p_best_rate = para.p_best_rate;
    pop_size = para.pop_size;
    arc_size = round(pop_size*para.arc_rate);
    nfes = 0;
    
    lu = [para.Xmin*ones(1,para.dim); para.Xmax*ones(1,para.dim)];
    pop = repmat(lu(1, :), pop_size, 1) + rand(pop_size, para.dim) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
    

    fitness = cec17_func( pop' , problem_index);
    nfes=nfes+pop_size;
    [bsf_fitness,bsf_index]=min(fitness);
    

    archive =[];
    memory_sf = 0.5*ones(para.memo_size, 1);
    memory_cr = 0.5*ones(para.memo_size, 1); 
    
    memory_pos = 1;
    

    pop_cr=[];pop_sf=[];
    p_num = round(pop_size*p_best_rate);
    
    max_pop_size = pop_size;
    min_pop_size = 4;
    arc_ind_count = 0;
    iter = 0;

    while nfes<para.maxfe 
        iter=iter+1;
        if iter == 50
            bsf_fitness_50iter = bsf_fitness;
        end

        success_sf = [] ; % S_F
        success_cr = []  ;% S_CR
        dif_fitness = []  ;% △fk
        [~,sorted_array] = sort(fitness);
        
        [a,b]=size(pop);
        child = ones(a,b);
        
        for target=1:pop_size
            random_selected_period = unidrnd(para.memo_size);

            mu_sf = memory_sf(random_selected_period);
            mu_cr = memory_cr(random_selected_period);
            
            if mu_cr == -1
                pop_cr(target) = 0;
            else
                pop_cr(target) = randn(1,1)*0.1+mu_cr;
                if pop_cr(target) > 1
                    pop_cr(target) = 1;
                elseif pop_cr(target) < 0
                    pop_cr(target)= 0;
                end
            end

            pop_sf(target) = cauchy_g(mu_sf, 0.1);
            while pop_sf(target) <= 0
                pop_sf(target) = cauchy_g(mu_sf, 0.1);  %直到pop_sf[target]>0
            end
            if pop_sf(target) > 1
               pop_sf(target) = 1.0;
            end
            
            p_best_ind = sorted_array(unidrnd(p_num));
            %operate current-to-pbest-1-bin-withArchive
            child(target,:) = current2pbest(pop, target, p_best_ind, pop_sf(target), pop_cr(target), archive, arc_ind_count, nfes, arc_size, pop_size,para);
             %pop, target, p_best_individual, scaling_factor, cross_rate, archive, arc_ind_count, nfes, arc_size, pop
        end
    
        fitness_child = cec17_func( child' , problem_index);
        for i=1:pop_size

            nfes = nfes+1;
            if (fitness_child(i)<bsf_fitness)
                bsf_fitness = fitness_child(i);
            end
            if nfes>=para.maxfe 
                break
            end
        end
                 if nfes-0.01*para.maxfe<pop_size
                 processvalue(1)=bsf_fitness;
                 elseif nfes-0.02*para.maxfe<pop_size
                 processvalue(2)=bsf_fitness; 
                 elseif nfes-0.03*para.maxfe<pop_size
                 processvalue(3)=bsf_fitness;
                 elseif nfes-0.05*para.maxfe<pop_size
                 processvalue(4)=bsf_fitness;
                 elseif nfes-0.1*para.maxfe<pop_size
                 processvalue(5)=bsf_fitness;
                 elseif nfes-0.2*para.maxfe<pop_size
                 processvalue(6)=bsf_fitness;
                 elseif nfes-0.3*para.maxfe<pop_size
                 processvalue(7)=bsf_fitness;
                 elseif nfes-0.4*para.maxfe<pop_size
                 processvalue(8)=bsf_fitness;
                 elseif nfes-0.5*para.maxfe<pop_size
                 processvalue(9)=bsf_fitness;
                 elseif nfes-0.6*para.maxfe<pop_size
                 processvalue(10)=bsf_fitness;
                 elseif nfes-0.7*para.maxfe<pop_size
                 processvalue(11)=bsf_fitness;
                 elseif nfes-0.8*para.maxfe<pop_size
                 processvalue(12)=bsf_fitness;
                 elseif nfes-0.9*para.maxfe<pop_size
                 processvalue(13)=bsf_fitness;
                 elseif nfes-1*para.maxfe<pop_size
                 processvalue(14)=bsf_fitness;
            end


        
        
        %%%%generation alternation
        for i=1:pop_size
            if fitness_child(i) == fitness(i)
                fitness(i) = fitness_child(i);
                pop(i,:) = child(i,:);
            elseif fitness_child(i) < fitness(i)
                dif_fitness=[dif_fitness,abs(fitness(i)-fitness_child(i))];
                %successful parameters are preserved in S_F and S_CR
                fitness(i) = fitness_child(i);
                pop(i,:) = child(i,:);                
                success_sf=[success_sf,pop_sf(i)];
                success_cr=[success_cr,pop_cr(i)];
                %parent vectors x_i which were worse than the trial vectors u_i are preserved
                if arc_size > 1
                    if arc_ind_count < arc_size 
                        archive(arc_ind_count+1,:) = pop(i,:);
                        arc_ind_count = arc_ind_count+1;
                    else  %Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted 
                        random_selected_arc_ind = unidrnd(arc_size);
                        archive(random_selected_arc_ind,:) = pop(i,:);
                    end
                end
                
            end
        end
        
        %update the bsf-solution
        
%         if nfes >= MAXFE_T && nfes < MAXFE_T + POP_SIZE
%             bsf_fitness_50tnfe = min(fitness);
%         end
%         if nfes >= MAXFE || bsf_fitness-OPTIMUM<=EPSILON 
%             break
%         end

        %old_num_success_params = num_success_params
        num_success_params = length(success_sf);
        
        %if numeber of successful parameters > 0, historical memories are updated 非空
        if num_success_params > 0
            
            memory_sf(memory_pos) = 0;
            memory_cr(memory_pos) = 0;
            temp_sum_sf = 0;
            temp_sum_cr = 0;
            sum = 0;
                %%%%%这里更改权重计算方式 
              RL_fitness=RL(dif_fitness,net_w,hidden_num);
            for i=1:num_success_params
                sum=sum+RL_fitness(i);
            end
            
            for i=1:num_success_params
%                 weight = dif_fitness(i) / sum;
                weight = RL_fitness(i) / sum;
                memory_sf(memory_pos) = memory_sf(memory_pos)+weight * success_sf(i) * success_sf(i);
                temp_sum_sf = temp_sum_sf+weight * success_sf(i);
                memory_cr(memory_pos) = memory_cr(memory_pos)+weight * success_cr(i) * success_cr(i);
                temp_sum_cr = temp_sum_cr+weight * success_cr(i);
            end
            memory_sf(memory_pos) = memory_sf(memory_pos)/temp_sum_sf;
            if temp_sum_cr == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = memory_cr(memory_pos)/temp_sum_cr;
            end

            memory_pos = memory_pos+1;
            if memory_pos > para.memo_size 
                memory_pos = 1; 
            end
            clear success_sf
            clear success_cr
            clear dif_fitness
        end
        
        %calculate the population size in the next generation
        plan_pop_size = round((((min_pop_size - max_pop_size) / para.maxfe) * nfes) + max_pop_size);
        if pop_size > plan_pop_size
            reduction_ind_num = int8(pop_size - plan_pop_size);
            if pop_size - reduction_ind_num < min_pop_size
                reduction_ind_num = int(pop_size - min_pop_size);
            end
            [pop, fitness, pop_size ]= reducePS(pop, fitness, reduction_ind_num, pop_size);
            arc_size = round(pop_size*para.arc_rate); 
            if arc_ind_count > arc_size
                arc_ind_count = arc_size;
            end

            p_num = round(pop_size*p_best_rate);
            if p_num <= 1
                p_num = 2;
            end
        end
       
       po(iter)=pop_size;
    end
end

function [child]=current2pbest(pop, target, p_best_individual, scaling_factor, cross_rate, archive, arc_ind_count, nfes, arc_size, pop_size,para)
    child = ones(para.dim,1);
    
    r1 = unidrnd(pop_size);
    r2 = unidrnd(pop_size+arc_ind_count);
    while r1 == target
        r1 = unidrnd(pop_size);
    end
    
    while r2 == target || r2 == r1
        r2 = unidrnd(pop_size+arc_ind_count);
    end
    
    random_variable =unidrnd(para.dim);
    
    %%%%mutation crossover
    if r2 > pop_size
        r2 = r2-pop_size;
        
        for i=1:para.dim
            if rand(1) < cross_rate || i == random_variable
                child(i) = pop(target, i) + scaling_factor * (pop(p_best_individual, i) - pop(target, i)) + scaling_factor * (pop(r1, i) - archive(r2, i));
            else
                child(i) = pop(target, i);
            end
        end
        
    else
        for i=1:para.dim
            if rand(1) < cross_rate || i == random_variable
                child(i) = pop(target, i) + scaling_factor * (pop(p_best_individual, i) - pop(target, i)) + scaling_factor * (pop(r1, i) - pop(r2, i));
            else
                child(i) = pop(target, i);
            end
        end    
    end
    child = modifySolutionWithParent(child , pop(target,:),para);
end

function [child] = modifySolutionWithParent(child, parent, para)
    for j=1:para.dim
        if child(j) < para.Xmin
            child(j)  = (para.Xmin + parent(j) )/2.0;
        elseif child(j)  > para.Xmax
            child(j)  = (para.Xmax + parent(j) )/2.0;
        end
    end
end

function [pop, fitness, pop_size] = reducePS(pop, fitness, reduction_ind_num, pop_size) %%%%%减小种群个数的时候删除点的函数
    for i=1:reduction_ind_num
        worst_ind = 1;
        for j=1:pop_size
            if fitness(j) > fitness(worst_ind)
                worst_ind = j;
            end 
        end
        pop(worst_ind,:)=[];
        fitness(worst_ind)=[];
        pop_size = pop_size-1;
    end
end

function [sam]=cauchy_g(mu, gamma)
    sam=mu + gamma * tan(pi * (rand(1) - 0.5));
end

function [act] = RL(F,net_w,hidden_num)
            F=(F)/max(F);
            hidnet=cell(1,hidden_num);
            hid=cell(1,hidden_num);
            output=cell(1,hidden_num);
            for j=1:hidden_num
                 hidnet{j}=F*net_w.layer{1}(j)+net_w.bias{1}(j);
                 hid{j}=(1+exp(-hidnet{j})).^(-1)-0.5;
            end
            net=net_w.bias{2};
            for j=1:hidden_num
                 net=net+hid{j}*net_w.layer{2}(j);
            end
            output=((1+exp(-net)).^(-1)-0.5)*15;

            act=output;
end
