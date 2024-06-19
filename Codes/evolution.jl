using Plots, ToeplitzMatrices
println("Start loading lib_fct.jl")
default(fmt = :png)    
include("lib_fct.jl");
println("End loading lib_fct.jl")

using Distances
using RDatasets, Clustering


# New function
# ==================================================================
# ==================================================================
# ==================================================================
function to_pad_vector(v, w_s)
    pv = zeros(w_s, 1)
    return vcat(pv,v)
end

function skiping_rate(y, dt)
    dt_w = 5 # taille de la fenetre en [ms]
    wind_size = convert(Int,round(dt_w/dt))
    dt_dw = 1 # taille du glissage en [ms]
    dw_size = convert(Int,round(dt_dw/dt))

    yp = to_pad_vector(y, wind_size)
    s_r = []
    for i in 1:dt_dw:length(yp)-wind_size
        ns = sum(yp[i:i+wind_size])
        push!(s_r,ns/(1000*wind_size))
    end
    return s_r
end

function give_score(reality, estimation, dt, score_type)
    if score_type == "MSE"
        sr_r = skiping_rate(reality,dt)
        sr_e = skiping_rate(estimation,dt)
        max_srr = maximum(sr_r)^2
        mse = (sr_r .- sr_e).^2
        mse_star = (sum(mse))/(length(reality)*dt)
        return mse_star
    elseif score_type == "TriFct"
        fct_tr_real = make_tr_filter(reality,dt)
        fct_tr_esti = make_tr_filter(estimation,dt)
        sc_tr = sum(abs.(fct_tr_real.-fct_tr_esti))
        return sc_tr
    end
end

function make_tr_filter(spike,dt)
    
    filtre_comp = copy(spike)
    filtre_comp = filtre_comp
    base_tri = 20
    n = dt/base_tri
    for i in 1:length(filtre_comp)-1
        if filtre_comp[i] > 0 && filtre_comp[i+1]<filtre_comp[i]
            filtre_comp[i+1] = filtre_comp[i]-n
        end
    end
    for i in length(filtre_comp):-1:2
        if filtre_comp[i] > 0 && filtre_comp[i-1]<filtre_comp[i]
            filtre_comp[i-1] = filtre_comp[i]-n
        end
    end
    return filtre_comp
end

function LL_AIC(yt, lambdat, n_param, score_type)

    if minimum(lambdat) <= 0
        A = copy(lambdat)
        M = maximum(A)
        A[A.<=0] .= M
        m = minimum(A)
        lambdat[lambdat.<=0] .= m
    end
    if isinf(maximum(lambdat))
        A = copy(lambdat)
        M = 0
        A[isinf.(A)] .= M
        m = maximum(A)
        lambdat[isinf.(lambdat)] .= m
    end
    LOG = log.(10,lambdat)
    LL = sum(yt .* LOG) - sum(lambdat)
    if isinf(LL)
        LL = -maximum(lambdat)
    end
    AIC = 2*(n_param - LL)
    if isinf(AIC)
        AIC = -LL
    end

    if score_type == "LL"
        return LL
    elseif score_type == "AIC"
        return AIC
    else
        return 0
    end
end

function deviance(reality, estimation, dt, r)
    prob_pi = (1 .-exp.(-r[:]./(1000/dt)))
    L1 = log10.(prob_pi)
    L2 = log10.(1 .-prob_pi)
    for i in 1:length(prob_pi)
        if abs(L1[i]) == Inf
            L1[i] = eps()*sign(L1[i])
        end
        if abs(L1[i]) == NaN
            L1[i] = eps()
        end
        if abs(L2[i]) == Inf
            L2[i] = eps()*sign(L2[i])
        end
        if abs(L1[i]) == NaN
            L2[i] = eps()
        end
    end
    deviance = sum(reality.*(L1) + (1 .-reality).*L2)
    return deviance
end

function generate_nice_HM(aX, aY, S, xname, yname, return_Mat=0)
    X = [] # Number of bases
    Y = [] # Number of training pariod
    for i in range(1,length(aX))
        if !("$(aX[i])" in X)
            push!(X,"$(aX[i])")
        end
    end
    for i in range(1,length(aY))
        if !("$(aY[i])" in Y)
            push!(Y,"$(aY[i])")
        end
    end
    S_Mat = zeros(length(Y), length(X))
    global z = 1 
    for i in range(1,length(X)), j in range(1,length(Y))
        S_Mat[j,i] = S[z]
        global z = z + 1
    end 

    gr()
    P = heatmap(X, Y, S_Mat, color=:buda, xlabel=latexstring(replace(xname, " " => "~")), ylabel=latexstring(replace(yname, " " => "~")) , xtickfontsize=12, ytickfontsize=12,  xlabelfontsize=16, ylabelfontsize=16 , right_margin=10Plots.mm)

    if return_Mat == 0
        return P
    else
        return P, S_Mat
    end
end

function SpDist_fct(real, esti, dt)
    L = length(real)
    global diff = 0
    for i in 1:L
        if !(real[i] == esti[i])
            if real[i] == 1
                First = real
                Second = esti
            else
                First = esti
                Second = real
            end
            for j in i+1:L
                if First[i] == Second[j]
                    break
                else
                    global diff = diff + 1
                end
            end
        end
    end
end

function lambda(dt, r)
    refreshRate = 1000/dt
    lambdat = (1 .-exp.(-r./refreshRate))
    for i in 1:length(lambdat)
        if lambdat[i] < eps()
            lambdat[i] = eps()
        end
    end
    return lambdat
end

function plot_spike_map(spikes, y, dt, T) 
    gr()
    PLT = plot()
    L = size(y, 2)
    plot!([dt,L*dt],[1,1],color=:white,label="")
    leg = 0
    for i in 1:L+1
        if i != L+1 # Plot GLM estimation
            for j in 1:size(y, 1)
                if y[j, i] == 1
                    if leg == 0
                        plot!([j, j]*dt, [i-0.35, i+0.35], color=:gray, linewidth=2, alpha=0.7, label="GLM responses")
                        leg = 1
                    else
                        plot!([j, j]*dt, [i-0.35, i+0.35], color=:gray, linewidth=2, alpha=0.7, label="")
                    end
                end
            end
        else # Plot Izhikevich sequence
            for j in 1:size(y, 1)
                if spikes[j] == 1
                    if leg == 1
                        plot!([j, j]*dt, [i-0.35, i+0.35], color=:black, linewidth=2, alpha=0.7, label="Izhikevich response")
                        leg = 0
                    else
                        plot!([j, j]*dt, [i-0.35, i+0.35], color=:black, linewidth=2, alpha=0.7, label="")
                    end
                end
            end
        end
    end
    plot!(ylabel = "Responses [trials] ", xlabel="Time [ms]", xlims = (T[1],T[2])) 
    return PLT
end

function find_best_idx(cell,nbreBasis, fct_fit, score_name, minMAX)
    """
    minMAX = -1  => min
    minMAX = +1  => max
    """
    global s_matrix = zeros(length(nbreBasis),length(nbreBasis))
    for n in nbreBasis, n2 in nbreBasis
        if score_name == "TriFct"
            s_matrix[n-nbreBasis[1]+1,n2-nbreBasis[1]+1] = mean(read_data("SavingOfComputation/DataOfCell_$(cell)/1_Training_Period/$(n)_basisK/$(n2)_basisH/$(fct_fit)/$(score_name)Score.csv"))
        else
            s_matrix[n-nbreBasis[1]+1,n2-nbreBasis[1]+1] = mean(read_data("SavingOfComputation/DataOfCell_$(cell)/AnalysisSolverData/1_Training_Period/$(n)_basisK/$(n2)_basisH/$(fct_fit)/Solv_value.csv"))
        end
    end 
    global s_matrix_with_3p3filter = zeros(length(nbreBasis)-2,length(nbreBasis)-2)
    for n in 1:length(nbreBasis)-2, n2 in 1:length(nbreBasis)-2
        s_matrix_with_3p3filter[n,n2] = sum(s_matrix[n:n+2,n2:n2+2])
    end 
    if minMAX == -1
        i_k_red = argmin(s_matrix_with_3p3filter)[2]
        i_h_red = argmin(s_matrix_with_3p3filter)[1]
    elseif minMAX == 1
        i_k_red = argmax(s_matrix_with_3p3filter)[2]
        i_h_red = argmax(s_matrix_with_3p3filter)[1]
    else
        println("Error in find_best_idx")
        return 0
    end 
    best_base_k = nbreBasis[i_k_red+1]
    best_base_h = nbreBasis[i_h_red+1]
    return [best_base_k, best_base_h]
end
# ==================================================================
# ==================================================================
# ==================================================================




global Ts = [400 1100;
    400 1250;
    400 1100;
    400 1250;
    100 1250;
    350 1250;
    9250 15250;
    8250 14250;
    750 1050;
    200 1100; # 10
    1900 4000;
    5100 9100;
    600 1500;
    600 1500;
    1400 2200; # 15
    10 100;
    200 1100;
    300 2200;
    200 1100;
    200 1100; #20
    550 850;
    100 1000;
    100 10000];
global Periods = [1000;
                1000;
                1000;
                1000;
                1000;
                1000;
                -1;
                -1; # 8
                -1;
                -1;
                -1;
                -1;
                -1;
                -1;
                -1;
                -1;
                -1;
                -1;
                -1;
                -1; # 20
                -1]

cell_num = [1,2,3,4]



nbreBasis = 3:1:14
nbre_training_Period = 1:1:8
runGLM = 10
global step = 1
score_type_all = ["TriFct"]
LLAICf = ["LLfit","AICfit"]
type_basis = ["","Denis"]

lk1 = ReentrantLock()
global lk_access_and_create_folder = ReentrantLock()

"""
    NOTE : 
        Some of the following 'if condition' are not used.
        But I decide to let it for your own use.
        As some 'if condition' are not used in the final version of the paper :
            "Modelling and classification of neuronal dynamics through Generalised Linear Models",
            I dont guaranty the good implementation. The checked 'if condition' are noted by a "G"
"""

# Fit the model
run_fit = false # G

# Generate HeatMap of the loss function of the GLM :--> TrainingPeriod VS NbreBasis
runLLf_fTP = false # G

# Generate HeatMap of the iteration of the GLM :--> TrainingPeriod VS NbreBasis
runIT_fTP = false # G

# Run the fitted model, generate runGLM trials,
run_generate = false # G

# Generate HeatMap of the loss function of the GLM
runLLf = false

# Genration HeatMap of the score of the simulation, plot those trials (5 first by default)
run_HM = false 

# Hold on h-filter (linked to bestK)
HolfOn_HF_for_Bestnbre_k = false

# Generate comparison graph of the loss function of fitted GLM (article and Denis)
run_comp_Basis1 = false # G

# Generate comparison graph of the score function of simulated GLM (article and Denis)
run_comp_Basis2 = false # G

# Find the best set of basis   --   Plot a supperposition of filter
run_plotFilter1 = false 

# Plot the best filter with a variation of the others number of basis
run_plotFilter2 = false

# Quantification of the filter deformation with the MSE on the curves
run_quantification_deformation = false

# Make HeatMap with the score of GLM, its gradient, its number of iteration   --   Plot the score of basis in the article and Denis' basis
run_HM_GLMscore = false  # G

# Plot k and h filters
Plot_best_filters = false  # G



global step = 1
if run_fit
    println("Is in 'if run_fit'")
    @time begin
        for cell in cell_num
            for n_tP in nbre_training_Period, nbasK in nbreBasis
            Threads.@threads for nbasH in nbreBasis
            for fct_fit in LLAICf
                lock(lk1) do
                    println(step, " / ", length(cell_num)*length(nbre_training_Period)*(length(nbreBasis)^2)*length(LLAICf))
                    global step = step + 1
                end
                if nbasK != nbasH && n_tP!=nbre_training_Period[1]
                    continue
                end
                for TB in type_basis
                    if TB == type_basis[2] && ( nbasH != 7 )
                        continue
                    end
                    # basis functions for stimulus filter
                    nkt = 100; # number of ms in stim filter
                    kbasprs = Dict(
                        :neye => 0,  # number of "identity" basis vectors near time of spike
                        :ncos => nbasK,  # number of raised-cosine vectors to use
                        :kpeaks => [0.1, round(nkt / 1.2)],  # position of first and last bump relative to identity bumps
                        :b => 10  # how nonlinear to make spacings (larger values make it more linear)
                    )
                    # basis functions for post-spike kernel
                    ihbasprs = Dict(
                        :ncols => nbasH,  # number of basis vectors for post-spike kernel
                        :hpeaks => [0.1, 100],  # peak location for the first and last vectors, in ms
                        :b => 10,  # how nonlinear to make spacings (larger values make it more linear)
                        :absref => 1  # absolute refractory period, in ms
                    )

                    maxIter = 1000;  # max number of iterations for fitting, also used for maximum number of function evaluations(MaxFunEvals)
                    tolFun = 1e-12;  # function tolerance for fitting                
                    
                    I1, dt1, T1, title1, time1 = generate_stimulation(cell)
                    Lm = convert(Int,(Ts[cell,2]+Periods[cell]*(n_tP-1))/dt1)
                    if Lm <= length(I1)
                        I1 = I1[1:Lm]
                        time1 = time1[1:Lm]
                        I_small = I1[1:convert(Int,(Ts[cell,2])/dt1)]
                    else
                        println("Error with the size of training period")
                    end
                    v1, u1, spikes1 = simulate_izhikevich(cell,I1,dt1);
                    ~, ~, spikes1_small = simulate_izhikevich(cell,I_small,dt1);
                    
                    if true
                        if nbasK == nbreBasis[1] && nbasH == nbreBasis[1]
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"I1.csv",I1)
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"dt1.csv",dt1)
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"T1.csv",T1)
                            save_title( joinpath( "SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period","title1.csv"  ),  title1)
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"time1.csv",time1)
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"v1.csv",v1)
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"u1.csv",u1)
                            save_data(["SavingOfComputation","DataOfCell_$(cell)","$(n_tP)_Training_Period"],"spikes1.csv",spikes1)        
                        end
                        solver = [NewtonTrustRegion()]
                        L = length(solver)
                        k, h, dc, prs, kbasis, hbasis = [],[],[],[],[],[]
                        Solv_value, Solv_Gnorm, Solv_metaData = [],[],[]
                        for i in 1:L
                            k_i, h_i, dc_i, prs_i, kbasis_i, hbasis_i, trace_i = fit_glm(I1,spikes1,dt1,nkt,kbasprs,ihbasprs,[],maxIter,tolFun,solver[i], TB, fct_fit);
                            push!(k, k_i), push!(h, h_i), push!(dc, dc_i), push!(prs, prs_i), push!(kbasis, kbasis_i), push!(hbasis, hbasis_i)
                            for j in 1:length(trace_i)
                                push!(Solv_value, trace_i[j].value), push!(Solv_Gnorm, trace_i[j].g_norm), push!(Solv_metaData, trace_i[j].metadata["time"])
                            end
                        end
                        k, h, dc, prs, kbasis, hbasis  = k[1], h[1], dc[1], prs[1], kbasis[1], hbasis[1];
                    
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"k$(TB).csv",k)
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"h$(TB).csv",h)
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"dc$(TB).csv",dc)
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"prs$(TB).csv",prs)
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"kbasis$(TB).csv",kbasis)
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"hbasis$(TB).csv",hbasis)

                        if fct_fit == LLAICf[1]
                            Solv_value = -Solv_value
                        end
                    
                        save_data(["SavingOfComputation","DataOfCell_$cell","AnalysisSolverData","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"Solv_value$(TB).csv",Solv_value)
                        save_data(["SavingOfComputation","DataOfCell_$cell","AnalysisSolverData","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"Solv_Gnorm$(TB).csv",Solv_Gnorm)
                        save_data(["SavingOfComputation","DataOfCell_$cell","AnalysisSolverData","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"Solv_metaData$(TB).csv",Solv_metaData)
                    end
                end
            end
            end
            end
        end
    end
end

if runLLf_fTP
    println("Is in 'if runLLf_fTP'")
    @time begin
        PLT = []
        Score_cell = zeros(length(nbre_training_Period), length(cell_num))
        fct_fit = LLAICf[1]
        for cell in cell_num
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","HeatMap"], "", "", 1)
            S = []
            P = []
            B = []
            for nbas in nbreBasis, n_tP in nbre_training_Period
                s = read_data("SavingOfComputation/DataOfCell_$cell/AnalysisSolverData/$(n_tP)_Training_Period/$(nbas)_basisK/$(nbas)_basisH/$(fct_fit)/Solv_value.csv")[end]
                s = s / n_tP
                push!(S,s)
                push!(P,n_tP)
                push!(B,nbas)
            end     
            S =  S./S[1]
            return_Mat = 1
            HM, S_Mat = generate_nice_HM(B, P, S, "n_k, n_h", "n_p", return_Mat)
            savefig(HM, joinpath(folder_path, "Score_BASISvsPERIOD.pdf"))

            c = findall(x->x==cell, cell_num)[1]
            Score_cell[:,c] = mean(S_Mat,dims=2)
            Score_cell[:,c] /= Score_cell[1,c]
        end

        gr()
        P = plot(xlabel=L"n_p", ylabel=L"LL", xlabelfontsize=16, ylabelfontsize=16)
        default_colors = palette(:lightrainbow, length(cell_num))
        for cell in cell_num
            ~, ~, ~, title, ~ = generate_stimulation(cell)
            c = findall(x->x==cell, cell_num)[1]
            id = findfirst(x -> x == cell, cell_num)
            plot!(Score_cell[:,c], color=default_colors[id], linestyle=:dot, label="", legendfontsize=12, xtickfontsize=12, ytickfontsize=12, xticks=1:length(Score_cell[:,c]))
            scatter!(Score_cell[:,c], color=default_colors[id], label=latexstring(replace(title, " " => "~")))
        end
        savefig(P, joinpath("SavingOfComputation","Figure", "Score_BASISvsPERIOD.pdf"))
    end
end

if runIT_fTP
    println("Is in 'if runIT_fTP'")
    @time begin
        PLT = []
        Score_cell = zeros(length(nbre_training_Period), length(cell_num))
        fct_fit = LLAICf[1]
        for cell in cell_num
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","HeatMap"], "", "", 1)
            S = []
            P = []
            B = []
            for nbas in nbreBasis, n_tP in nbre_training_Period
                s = length(read_data("SavingOfComputation/DataOfCell_$cell/AnalysisSolverData/$(n_tP)_Training_Period/$(nbas)_basisK/$(nbas)_basisH/$(fct_fit)/Solv_value.csv"))
                push!(S,s)
                push!(P,n_tP)
                push!(B,nbas)
            end     
            S =  S./S[1]
            return_Mat = 1
            HM, S_Mat = generate_nice_HM(B, P, S, "Number of basis for filters k and h", "Number of periods", return_Mat)
            savefig(HM, joinpath(folder_path, "IT_BASISvsPERIOD.pdf"))

            c = findall(x->x==cell, cell_num)[1]
            Score_cell[:,c] = mean(S_Mat,dims=2)
            Score_cell[:,c] /= Score_cell[1,c]
        end
        
        gr()
        P = plot(xlabel=L"Number~of~periods", ylabel=L"Number~of~iterations", xlabelfontsize=16, ylabelfontsize=16)
        default_colors = palette(:lightrainbow, length(cell_num))
        for cell in cell_num
            ~, ~, ~, title, ~ = generate_stimulation(cell)
            c = findall(x->x==cell, cell_num)[1]
            id = findfirst(x -> x == cell, cell_num)
            plot!(ylims=(0.4,2.5),Score_cell[:,c],color=default_colors[id], linestyle=:dot, label="", legendfontsize=12, xtickfontsize=12, ytickfontsize=12, xticks=1:length(Score_cell[:,c]), legend=:best)
            scatter!(Score_cell[:,c], color=default_colors[id], label=latexstring(replace(title, " " => "~")))
        end
        savefig(P, joinpath("SavingOfComputation","Figure", "IT_BASISvsPERIOD.pdf"))
    end
    @time begin
        PLT = []
        Score_cell = zeros(length(nbre_training_Period), length(cell_num))
        fct_fit = LLAICf[1]
        for cell in cell_num
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","HeatMap"], "", "", 1)
            S = []
            P = []
            B = []
            for nbas in nbreBasis, n_tP in nbre_training_Period
                s = read_data("SavingOfComputation/DataOfCell_$cell/AnalysisSolverData/$(n_tP)_Training_Period/$(nbas)_basisK/$(nbas)_basisH/$(fct_fit)/Solv_metaData.csv")[end]
                push!(S,s)
                push!(P,n_tP)
                push!(B,nbas)
            end     
            S =  S./S[1]
            return_Mat = 1
            HM, S_Mat = generate_nice_HM(B, P, S, "Number of basis for filters k and h", "Number of periods", return_Mat)
            savefig(HM, joinpath(folder_path, "Time_BASISvsPERIOD.pdf"))

            c = findall(x->x==cell, cell_num)[1]
            Score_cell[:,c] = mean(S_Mat,dims=2)
            Score_cell[:,c] /= Score_cell[1,c]
        end
        
        gr()
        P = plot(xlabel=L"Number~of~periods", ylabel=L"Normed~time")
        for cell in cell_num
            ~, ~, ~, title, ~ = generate_stimulation(cell)
            c = findall(x->x==cell, cell_num)[1]
            plot!(Score_cell[:,c], linewidth=3, label=latexstring(replace(title, " " => "~")))
        end
        savefig(P, joinpath("SavingOfComputation","Figure", "Time_BASISvsPERIOD.pdf"))
    end
end

global step = 1
if run_generate
    println("Is in 'if run_generate'")
    @time begin
        for cell in cell_num
            I1, dt1, T1, title1, time1 = generate_stimulation(cell)
            for n_tP in nbre_training_Period, nbasK in nbreBasis, nbasH in nbreBasis, fct_fit in LLAICf
                lock(lk1) do
                    println(step, " / ", length(nbre_training_Period)*length(cell_num)*(length(nbreBasis)^2)*length(LLAICf))
                    global step = step + 1
                end
                if nbasK != nbasH && n_tP!=nbre_training_Period[1]
                    continue
                end 

                Lm = convert(Int,(Ts[cell,2]+Periods[cell]*(n_tP-1))/dt1) # Trick for speed up the generation (run on 1 period)
                if Lm <= length(I1)
                    I_small = I1[1:Lm]
                else
                    println("Error with the size of training period")
                    println("I1 length ", length(I1))
                    println("Lm value ", Lm)
                    println("n_tP value ", n_tP)
                    return 0
                end
                ~, ~, spikes1_small = simulate_izhikevich(cell,I_small,dt1);

                # Simulation of the GLM
                for TB in type_basis
                    if (TB == type_basis[2] && ( nbasH != 7 )) || (TB == type_basis[2] && n_tP!=nbre_training_Period[1])
                        continue
                    end
                    prs = read_data("SavingOfComputation/DataOfCell_$cell/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/prs$(TB).csv")
                    prs_k = prs[1:nbasK]
                    prs_h = prs[nbasK+1:end-1]

                    kbasis = read_data("SavingOfComputation/DataOfCell_$cell/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/kbasis$(TB).csv")
                    hbasis = read_data("SavingOfComputation/DataOfCell_$cell/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/hbasis$(TB).csv")

                    k = kbasis * prs_k
                    h = hbasis * prs_h
                    dc = prs[end]

                    y1, stimcurr1, hcurr1, r1 = simulate_glm(I_small, dt1, k,h,dc,runGLM);
                    for t in 1:runGLM
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"y$(t)$(TB).csv",y1[:,t])
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"hcurr$(t)$(TB).csv",hcurr1[:,t])
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"r$(t)$(TB).csv",r1[:,t])
                    end
                    save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"stimcurr$(TB).csv",stimcurr1)

                    Sco = zeros(size(y1,2), length(score_type_all))
                    for i in 1:1:size(y1,2)
                        for score_type in score_type_all
                            if score_type == "TriFct"
                                s = give_score(spikes1_small[:], y1[:,i], dt1, score_type)
                            else
                                println("Error in the type of score ")
                                return 0
                            end
                                # if score_type == "DWT"
                                #     s = dtw(spikes1_small[:], y1[:,i])[1]
                                # elseif score_type == "FastDWT"
                                #     s = fastdtw(spikes1_small[:], y1[:,i], SqEuclidean(), convert(Int,round(25/dt1)) )[1]
                                # elseif score_type == "TriFct"
                                #     s = give_score(spikes1_small[:], y1[:,i], dt1, score_type)
                                # elseif score_type == "MSE"
                                #     s = give_score(spikes1_small[:], y1[:,i], dt1, score_type)
                                # elseif score_type == "LL"
                                #     s = LL_AIC(spikes1_small[:], r1[:,i], (nbasK+nbasH)+1, score_type)
                                # elseif score_type == "AIC"
                                #     s = LL_AIC(spikes1_small[:], r1[:,i], (nbasK+nbasH)+1, score_type)
                                # elseif score_type== "SpDist"
                                #     s = SpDist_fct(spikes1_small[:], y1[:,i], dt1)
                                # elseif score_type== "Deviance"
                                #     s = deviance(spikes1_small[:],  y1[:,i], dt1, r1[:,i])
                                # else
                                #     println("Error in the type of score ")
                                #     return 0
                                # end
                            Sco[i,findall(x->x==score_type, score_type_all)[1]] = s 
                                # if score_type == score_type_all[end]
                                #     Lmbd = 1
                                # end
                        end
                    end
                    for score_type in score_type_all
                        save_data(["SavingOfComputation","DataOfCell_$cell","$(n_tP)_Training_Period","$(nbasK)_basisK","$(nbasH)_basisH","$(fct_fit)"],"$(score_type)Score$(TB).csv",Sco[:,findall(x->x==score_type, score_type_all)])
                    end
                end
            end 
        end 
    end 
end 

global step = 1
if runLLf
    println("Is in 'if runLLf'")
    @time begin
        for cell in cell_num, fct_fit in LLAICf
            lock(lk1) do
                println(step, " / ", length(cell_num)*length(LLAICf))
                global step = step + 1
            end
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","HeatMap"], "", "", 1)
            Bh = []
            Bk = []
            Scofit = []
            Ratio = []
            for n_tP in [1], nbasK in nbreBasis, nbasH in nbreBasis
                file = "SavingOfComputation/DataOfCell_$(cell)/AnalysisSolverData/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/Solv_value.csv"
                if isfile(file)
                    Sfit = read_data(file)[end]
                else
                    Sfit = -1
                end
                push!(Bh, nbasH)
                push!(Bk, nbasK)
                push!(Scofit, Sfit)
            end
            Scofit = Scofit./Scofit[1]
            return_Mat = 1
            HM, S_Mat = generate_nice_HM(Bk, Bh, Scofit, L"n_k", L"n_h", return_Mat)
            Dif_Mat_S = S_Mat
            for i in 1:size(Dif_Mat_S,1), j in 1:size(Dif_Mat_S,2)
                if j == (1:size(Dif_Mat_S,2))[end]
                    Dif_Mat_S[i,j] = Dif_Mat_S[i,j-1]
                else
                    Dif_Mat_S[i,j] = abs((S_Mat[i,j]/S_Mat[i,j+1])-1)
                end
            end
            Red_Mat = maximum(Dif_Mat_S, dims=1)


            Red_Mat_Vector = []
            for i in 1:length(Red_Mat)
                push!(Red_Mat_Vector,Red_Mat[i])
            end

            # Check best option 
            Value_threshold_diff = 0.05
            for i in 7:-1:1
                if Red_Mat_Vector[i] > Value_threshold_diff
                    bb = nbreBasis[end]
                    if i != length(Red_Mat_Vector)
                        bb = nbreBasis[i+1]
                    end
                    save_data(["SavingOfComputation","DataOfCell_$(cell)","NumberOfBasisInFilter_K_for_analysis"],"nbre_k_$(fct_fit).csv",bb)
                    println("For cell ", cell, " bb is ", bb)
                    break
                end
            end
            P = plot(3:1:3-1+length(Red_Mat_Vector),Red_Mat_Vector, color=:blue, xlabel=L"n_k", ylabel=L"Maximum~difference~ratio", label="", xticks=nbreBasis[1]:1:nbreBasis[end],linestyle=:dot, xtickfontsize=12, ytickfontsize=12,  xlabelfontsize=16, ylabelfontsize=16)
            scatter!(3:1:3-1+length(Red_Mat_Vector),Red_Mat_Vector, color=:blue, ms=5,label="")
            savefig(P, joinpath(folder_path, "Score$(fct_fit)_Difference_Ratio.pdf"))

            savefig(HM, joinpath(folder_path, "Score$(fct_fit)_accordingNumOfBasis.pdf"))
            gr()
            savefig(plot(nbreBasis,Red_Mat_Vector,label="",xlabel=L"Number~of~basis",ylabel=L"Ratio~difference~with~the~following"), joinpath(folder_path, "RatioDiffMax_ofScore$(fct_fit)_accordingNumOfBasis.pdf"))
        end
    end
end

global step = 1
if run_HM
    println("Is in 'if run_HM'")
    for score_type in score_type_all
        for cell in cell_num
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","HeatMap"], "", "", 1)
            sp_IZ = read_data("SavingOfComputation/DataOfCell_$(cell)/1_Training_Period/spikes1.csv",1)
            dt = read_data("SavingOfComputation/DataOfCell_$(cell)/1_Training_Period/dt1.csv")[1]
            for fct_fit in LLAICf, n_tP in nbre_training_Period, nbasK in nbreBasis, nbasH in nbreBasis
                lock(lk1) do
                    println(step, " / ", length(nbre_training_Period)*length(cell_num)*(length(nbreBasis)^2)*length(LLAICf)*length(score_type_all))
                    global step = step + 1
                end
                if nbasK != nbasH && n_tP!=nbre_training_Period[1]
                    continue
                end
                s = median(read_data("SavingOfComputation/DataOfCell_$cell/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/$(score_type)Score.csv"))
                if nbasK == nbasH
                    if n_tP == nbre_training_Period[1] && nbasK == nbreBasis[1]
                        global NB = []
                        global NT = []
                        global S = []
                    end
                    push!(NB,nbasK)
                    push!(NT,n_tP)
                    push!(S,s)
                    if n_tP == nbre_training_Period[end] && nbasK == nbreBasis[end]
                        HM = generate_nice_HM(NB, NT, S, "Number of basis for k and h filters", "Period of training")
                        savefig(HM, joinpath(folder_path, "$(score_type)Score_BASISvsPERIOD.html"))
                    end
                end
                if n_tP == nbre_training_Period[1] 
                    if score_type == score_type_all[1]
                        for TB in type_basis
                            if TB == type_basis[2] && ( nbasH != 7 )
                                continue
                            end
                            yit = read_data("SavingOfComputation/DataOfCell_$(cell)/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/y1$(TB).csv",1)
                            y = zeros(length(yit),runGLM)
                            y[:,1] = yit
                            for t in 2:runGLM
                                yit = read_data("SavingOfComputation/DataOfCell_$(cell)/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/y$(t)$(TB).csv",1)
                                y[:,t] = yit
                            end

                            y = y[:,1:5] # Do a real raster plot !!
                            name_to_save = "TP$(n_tP)_with_$(nbasK)_basisK_$(nbasH)_basisH$(TB).pdf"
                            PLT = plot_spike_map(sp_IZ, y, dt, Ts[cell,:])
                            savefig(PLT, joinpath(folder_path, name_to_save))
                            PLT = []
                        end
                    end
                    if nbasK == nbreBasis[1] && nbasH == nbreBasis[1]
                        global NB_k = []
                        global NB_h = []
                        global NT_p = []
                        global S_all = []
                    end
                    push!(NB_k,nbasK)
                    push!(NB_h,nbasH)
                    push!(NT_p,n_tP)
                    push!(S_all,s)
                    if nbasK == nbreBasis[end] && nbasH == nbreBasis[end]
                        NBk = NB_k
                        NBh = NB_h
                        Sa = S_all
                        HM = generate_nice_HM(NBk, NBh, Sa, "Number of basis for k filters", "Number of basis for h filters")
                        savefig(HM, joinpath(folder_path, "$(score_type)Score_KvsH.html"))
                        HM = []
                    end
                end
            end 
        end 
    end
end

gr()
global step = 1
if run_comp_Basis1
    println("Is in 'if run_comp_Basis1'")
    for cell in cell_num
        folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
        for fct_fit in LLAICf
            lock(lk1) do
                println(step, " / ", length(cell_num)*length(LLAICf))
                global step = step + 1
            end
            if fct_fit == "LLfit" 
                ylab = L"LL"
            elseif fct_fit == "AICfit" 
                ylab = L"AIC"
            else
                ylab = score_type
            end
            P = plot(ylabel = ylab, xlabel=L"n_k",xlabelfontsize=16, ylabelfontsize=16)
            S1 = []
            for TB in type_basis
                S = []
                for n_tP in [1], nbasK in nbreBasis, nbasH in [7]
                    s = mean(read_data("SavingOfComputation/DataOfCell_$cell/AnalysisSolverData/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/Solv_value$(TB).csv"))
                    push!(S,s)
                end
                if TB == ""
                    global Sref = S[1]
                end
                if TB == ""
                    lab = L"Raised~cosines~basis"
                    col = "blue"
                elseif TB == "Denis"
                    lab = L"Homemade~basis"
                    col = "red"
                end
                plot!(nbreBasis, S./Sref, label="", linestyle=:dot ,color=col,xticks=nbreBasis[1]:1:nbreBasis[end],xtickfontsize=12, ytickfontsize=12, legendfontsize=12)
                scatter!(nbreBasis, S./Sref, ms=5, label=lab,color=col,legend=:best)
            end
            savefig(P, joinpath(folder_path, "Comp_with_$(fct_fit).pdf") )
        end
    end
end

global step = 1
if run_comp_Basis2
    println("Is in 'if run_comp_Basis2'")
    for cell in cell_num
        folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
        for score_type in score_type_all, fct_fit in LLAICf
            lock(lk1) do
                println(step, " / ", length(cell_num)*length(score_type_all)*length(LLAICf))
                global step = step + 1
            end
            if score_type == "LLfit" 
                ylab = L"LL"
            elseif score_type == "TriFct" 
                ylab = L"TrSc"
            else
                ylab = score_type
            end
            P = plot(ylabel = ylab, xlabel=L"n_k",xlabelfontsize=16, ylabelfontsize=16)
            for TB in type_basis
                S = []
                for n_tP in [1], nbasK in nbreBasis, nbasH in [7]
                    s = mean(read_data("SavingOfComputation/DataOfCell_$cell/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/$(score_type)Score$(TB).csv"))
                    push!(S,s)
                end
                if TB == ""
                    global Sref = S[1]
                end
                if TB == ""
                    lab = L"Raised~cosines~basis"
                    col = "blue"
                elseif TB == "Denis"
                    lab = L"Homemade~basis"
                    col = "red"
                end
                plot!(nbreBasis, S./Sref, label="", linestyle=:dot ,color=col,xticks=nbreBasis[1]:1:nbreBasis[end],xtickfontsize=12, ytickfontsize=12, legendfontsize=12)
                scatter!(nbreBasis, S./Sref, ms=5, label=lab,color=col)
            end
            savefig(P, joinpath(folder_path, "Comp2_with_$(score_type).pdf") )
        end
    end
end

if HolfOn_HF_for_Bestnbre_k
    println("Is in 'if HolfOn_HF_for_Bestnbre_k'")
    global step = 1
    fct_fit = LLAICf[1]
    for cell in cell_num, fct_fit in LLAICf
        dt = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/dt1.csv")[1]
        nb_K_best = convert(Int,read_data("SavingOfComputation/DataOfCell_$(cell)/NumberOfBasisInFilter_K_for_analysis/nbre_k_$(fct_fit).csv")[1])
        P = plot(xlabel=L"Time~[ms]",ylabel=L"Intensity")
        H = []
        Sc = []
        default_colors = palette(:lightrainbow, length(nbreBasis))
        for nbH in nbreBasis
            prs = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nb_K_best)_basisK/$(nbH)_basisH/$(fct_fit)/prs.csv")
            prs_h = prs[nb_K_best+1:end-1]
            basis_h = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nb_K_best)_basisK/$(nbH)_basisH/hbasis.csv")
            h_with_best_k = basis_h*prs_h
            Lab = "h_{$(nbH)}"
            x = (-1500:-1)*dt
            if length(h_with_best_k) >= 1500
                y = h_with_best_k[1:1500]
            else
                y = vcat(zeros(1500-length(h_with_best_k)),h_with_best_k)
            end
            push!(H,y)
            id = findfirst(x -> x == nbH, nbreBasis)
            plot!(x,reverse(y),label=latexstring("\\textrm{$(replace(Lab, " " => "~"))}"), color=default_colors[id],legendfontsize=12)
        end
        # DTW Analysis
        for i in 1:length(nbreBasis)
            if i > 1
                println("cell ", cell, " dtw between ", nbreBasis[i], " and ", nbreBasis[i-1])
                println("-- dtw is ", dtw(H[i][:],H[i-1][:])[1])
            end
            if i < length(nbreBasis)
                println("cell ", cell, " dtw between ", nbreBasis[i], " and ", nbreBasis[i+1])
                println(" -- dtw is ", dtw(H[i][:],H[i+1][:])[1])
            end
        end

        folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
        y_min = sort(minimum.(H))[2]
        y_max = maximum(maximum.(H))
        plot!(ylims=(y_min, y_max), xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
        savefig(P, joinpath(folder_path, "CompH_with_$(fct_fit)_with_bestK.pdf") )
    end
end


gr()
global step = 1
if run_plotFilter1
    println("Is in 'if run_plotFilter1'")
    global step = 1
    for cell in cell_num
        folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
        for fct_fit in LLAICf, score_name in vcat(score_type_all[:],""), nb in nbreBasis
            lock(lk1) do
                println(step, " / ", length(cell_num)*length(nbreBasis)*length(LLAICf)*(length(score_type_all)+1))
                global step = step + 1
            end
            bk = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nb)_basisK/$(nb)_basisH/$(fct_fit)/k.csv");
            bh = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nb)_basisK/$(nb)_basisH/$(fct_fit)/h.csv")
            if nb == nbreBasis[1]
                global BK = zeros(length(nbreBasis), length(bk))
                global BH = copy(BK)
            end
            BK[nb-nbreBasis[1]+1,:] = bk
            if length(bh) >= length(bk) # ok
                BH[nb-nbreBasis[1]+1,:] = bh[1:length(bk)]
            else
                BH[nb-nbreBasis[1]+1,:] = vcat(bh, zeros(length(bk)-length(bh),1) )
            end
            if nb == nbreBasis[end]
                dt = read_data("SavingOfComputation/DataOfCell_$(cell)/1_Training_Period/dt1.csv")[1]

                minMAX = -1
                if fct_fit == "LLfit" && score_name == ""
                    minMAX = 1
                end
                id_best_sc = find_best_idx(cell, nbreBasis, fct_fit, score_name, minMAX)
                save_data(["SavingOfComputation","DataOfCell_$cell","1_Training_Period","$(fct_fit)"],"best_basis$(score_name).csv",id_best_sc)
                best_K = id_best_sc[1]
                best_H = id_best_sc[2]

                Pk = plot()
                x_inv = [-size(BK,2)*dt:dt:-dt]
                for i in nbreBasis
                    if i != best_K
                        plot!(x_inv,BK[i-nbreBasis[1]+1,:],color="black", alpha=0.1*i, label="k-filter $(i)")
                    end
                end
                plot!(x_inv, BK[best_K-nbreBasis[1]+1,:],color="blue", label="k-filter $(best_K)")

                Ph = plot()
                for i in nbreBasis
                    if i != best_H
                        plot!([dt:dt:size(BH,2)*dt],BH[i-nbreBasis[1]+1,:],color="black", alpha=0.1*i, label="h-filter $(i)")
                    end
                end
                plot!([dt:dt:size(BH,2)*dt], BH[best_H-nbreBasis[1]+1,:],color="blue", label="h-filter $(best_H)")
                PLT = plot(Pk, Ph)
                savefig(PLT, joinpath(folder_path, "ComparisonFilter_using$(fct_fit)$(score_name).svg"))
            end 
        end
    end
end
    
global step = 1
if run_plotFilter2
    println("Is in 'if run_plotFilter2'")
    global step = 1
    for cell in cell_num
        for fct_fit in LLAICf, score_name in vcat(score_type_all[:],"")
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
            lock(lk1) do
                println(step, " / ", length(cell_num)*length(LLAICf)*(length(score_type_all)+1))
                global step = step + 1
            end
            nbK = convert(Int,read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(fct_fit)/best_basis$(score_name).csv")[1])
            nbH = convert(Int,read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(fct_fit)/best_basis$(score_name).csv")[2])
            dt = read_data("SavingOfComputation/DataOfCell_$(cell)/1_Training_Period/dt1.csv")[1]
            for nbre_B in nbreBasis
                h_with_best_k = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbK)_basisK/$(nbre_B)_basisH/$(fct_fit)/h.csv");
                k_with_best_h = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbre_B)_basisK/$(nbH)_basisH/$(fct_fit)/k.csv");
                best_k_with_h = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbK)_basisK/$(nbre_B)_basisH/$(fct_fit)/k.csv");
                best_h_with_k = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbre_B)_basisK/$(nbH)_basisH/$(fct_fit)/h.csv");

                if nbre_B == nbreBasis[1]
                    global BK = zeros(length(nbreBasis), length(k_with_best_h))
                    global BH = copy(BK)
                    global BK_2 = copy(BK)
                    global BH_2 = copy(BK)
                end

                BK[nbre_B-nbreBasis[1]+1,:] = k_with_best_h
                if length(h_with_best_k) >= length(k_with_best_h) # ok
                    BH[nbre_B-nbreBasis[1]+1,:] = h_with_best_k[1:length(k_with_best_h)]
                else
                    BH[nbre_B-nbreBasis[1]+1,:] = vcat(h_with_best_k, zeros(length(k_with_best_h)-length(h_with_best_k),1) )
                end

                BK_2[nbre_B-nbreBasis[1]+1,:] = best_k_with_h
                if length(best_h_with_k) >= length(best_k_with_h) # ok
                    BH_2[nbre_B-nbreBasis[1]+1,:] = best_h_with_k[1:length(best_k_with_h)]
                else
                    BH_2[nbre_B-nbreBasis[1]+1,:] = vcat(best_h_with_k, zeros(length(best_k_with_h)-length(best_h_with_k),1) )
                end

                if nbre_B == nbreBasis[end]
                    x_inv = [-size(BK,2)*dt:dt:-dt]
                    Pk = plot()
                    for i in nbreBasis
                        if i != nbK
                            plot!(x_inv,BK[i-nbreBasis[1]+1,:],color="black", alpha=0.1*i, label="k-filter $(i) with h-filter $(nbH)")
                        end
                    end
                    plot!(x_inv, BK[nbK-nbreBasis[1]+1,:],color="blue", label="k-filter $(nbK) with h-filter $(nbH)")

                    Ph = plot()
                    for i in nbreBasis
                        if i != nbH
                            plot!([dt:dt:size(BH,2)*dt],BH[i-nbreBasis[1]+1,:],color="black", alpha=0.1*i, label="h-filter $(i) with k-filter $(nbK)")
                        end
                    end
                    plot!([dt:dt:size(BH,2)*dt], BH[nbH-nbreBasis[1]+1,:],color="blue", label="h-filter $(nbH) with k-filter $(nbK)")

                    Pk_2 = plot()
                    for i in nbreBasis
                        if i != nbH
                            plot!(x_inv,BK_2[i-nbreBasis[1]+1,:],color="black", alpha=0.1*i, label="k-filter $(nbK) with h-filter $(i)")
                        end
                    end
                    plot!(x_inv, BK_2[nbH-nbreBasis[1]+1,:],color="blue", label="k-filter $(nbK) with h-filter $(nbH)")

                    Ph_2 = plot()
                    for i in nbreBasis
                        if i != nbK
                            plot!([dt:dt:size(BH_2,2)*dt],BH_2[i-nbreBasis[1]+1,:],color="black", alpha=0.1*i, label="h-filter $(nbH) with k-filter $(i)")
                        end
                    end
                    plot!([dt:dt:size(BH_2,2)*dt], BH_2[nbK-nbreBasis[1]+1,:],color="blue", label="h-filter $(nbH) with k-filter $(nbK)")

                    PLT = plot(Pk, Ph, Pk_2, Ph_2, size=(1000,1000))
                    savefig(PLT, joinpath(folder_path, "Comp_variation_using$(fct_fit)$(score_name).svg") )
                end
            end
        end
    end
end

global step = 1
if run_quantification_deformation
    println("Is in 'if run_quantification_deformation'")
    global step = 1
    for fct_fit in LLAICf, score_name in vcat(score_type_all[:],"")
        for cell in cell_num
            folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
            lock(lk1) do
                println(step, " / ", length(cell_num)*length(LLAICf)*(length(score_type_all)+1))
                global step = step + 1
            end

            dt = read_data("SavingOfComputation/DataOfCell_$(cell)/1_Training_Period/dt1.csv")[1]
            nbK = convert(Int,read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(fct_fit)/best_basis$(score_name).csv")[1])
            nbH = convert(Int,read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(fct_fit)/best_basis$(score_name).csv")[2])
            best_k = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbK)_basisK/$(nbH)_basisH/$(fct_fit)/k.csv");
            best_h = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbK)_basisK/$(nbH)_basisH/$(fct_fit)/h.csv");
            if length(best_h) >= length(best_k)
                best_h = best_h[1:length(best_k)-1]
            else
                best_h = vcat(best_h, zeros(length(best_k)-length(best_h),1) )
            end

            global MSE_1 = zeros(length(nbreBasis),1)
            global MSE_2 = zeros(length(nbreBasis),1)
            global MSE_3 = zeros(length(nbreBasis),1)
            global MSE_4 = zeros(length(nbreBasis),1)    
            for nbre_B in nbreBasis
                h_with_best_k = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbK)_basisK/$(nbre_B)_basisH/$(fct_fit)/h.csv");
                if length(h_with_best_k) >= length(best_h) # ok
                    h_with_best_k = h_with_best_k[1:length(best_h)]
                else
                    h_with_best_k = vcat(h_with_best_k, zeros(length(best_h)-length(h_with_best_k),1) )
                end

                k_with_best_h = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbre_B)_basisK/$(nbH)_basisH/$(fct_fit)/k.csv");

                best_k_with_h = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbK)_basisK/$(nbre_B)_basisH/$(fct_fit)/k.csv");

                best_h_with_k = read_data("SavingOfComputation/DataOfCell_$cell/1_Training_Period/$(nbre_B)_basisK/$(nbH)_basisH/$(fct_fit)/h.csv");
                if length(best_h_with_k) >= length(best_h) # ok
                    best_h_with_k = best_h_with_k[1:length(best_h)]
                else
                    best_h_with_k = vcat(best_h_with_k, zeros(length(best_h)-length(best_h_with_k),1) )
                end

                mse_diff_k_same_h = sum( (k_with_best_h .- best_k).^2 )/(length(best_k)*dt)
                mse_diff_h_same_k = sum( (h_with_best_k .- best_h).^2 )/(length(best_h)*dt)
                mse_same_k_diff_h = sum( (best_k_with_h .- best_k).^2 )/(length(best_k)*dt)
                mse_same_h_diff_k = sum( (best_h_with_k .- best_h).^2 )/(length(best_h)*dt)
                global MSE_1[nbre_B-nbreBasis[1]+1] = mse_diff_k_same_h
                global MSE_2[nbre_B-nbreBasis[1]+1] = mse_diff_h_same_k
                global MSE_3[nbre_B-nbreBasis[1]+1] = mse_same_k_diff_h
                global MSE_4[nbre_B-nbreBasis[1]+1] = mse_same_h_diff_k
                
                if nbre_B == nbK
                    global MSE_1[nbre_B-nbreBasis[1]+1] = NaN
                    global MSE_4[nbre_B-nbreBasis[1]+1] = NaN
                end

                if nbre_B == nbH
                    global MSE_2[nbre_B-nbreBasis[1]+1] = NaN
                    global MSE_3[nbre_B-nbreBasis[1]+1] = NaN
                end

                save_data(["SavingOfComputation","DataOfCell_$(cell)","Quantification_filter_MSE","$(fct_fit)$(score_name)"],"k$(nbre_B)VSkbest_with_bestH.csv",mse_diff_k_same_h)
                save_data(["SavingOfComputation","DataOfCell_$(cell)","Quantification_filter_MSE","$(fct_fit)$(score_name)"],"h$(nbre_B)VShbest_with_bestK.csv",mse_diff_h_same_k)
                save_data(["SavingOfComputation","DataOfCell_$(cell)","Quantification_filter_MSE","$(fct_fit)$(score_name)"],"kbestVSkbest_with_hbestAndh$(nbre_B).csv",mse_same_k_diff_h)
                save_data(["SavingOfComputation","DataOfCell_$(cell)","Quantification_filter_MSE","$(fct_fit)$(score_name)"],"hbestVShbest_with_kbestAndk$(nbre_B).csv",mse_same_h_diff_k)
            end
            PLT1 = scatter(nbreBasis, MSE_1, label="MSE per [ms] k-filter with best h-filter",yaxis=:log10)
            PLT2 = scatter(nbreBasis, MSE_2, label="MSE per [ms] h-filter with best k-filter",yaxis=:log10)
            PLT3 = scatter(nbreBasis, MSE_3, label="MSE per [ms] best k-filter with different h-filter",yaxis=:log10)
            PLT4 = scatter(nbreBasis, MSE_4, label="MSE per [ms] best h-filter with different k-filter",yaxis=:log10)
            PLT = plot(PLT1,PLT2,PLT3,PLT4, size=(1000,1000))
            savefig(PLT, joinpath(folder_path, "Quantification_with_$(fct_fit)$(score_name).svg") )
        end
    end
end

global step = 1
if run_HM_GLMscore
    println("Is in 'if run_HM_GLMscore'")
    for fct_fit in LLAICf
        for cell in cell_num
            Sc= zeros(length(nbreBasis),length(nbreBasis))
            ScG= zeros(length(nbreBasis),length(nbreBasis))
            Scit= zeros(length(nbreBasis),length(nbreBasis))

            ScD= zeros(length(nbreBasis),1)
            for TB in type_basis, nbasK in nbreBasis, nbasH in nbreBasis
                lock(lk1) do
                    println(step, " / ", length(cell_num)*length(LLAICf)*(length(nbreBasis)^2)*length(type_basis))
                    global step = step + 1
                end
                if TB == type_basis[2] && nbasH != 7
                    continue
                end
                s = read_data("SavingOfComputation/DataOfCell_$cell/AnalysisSolverData/1_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/Solv_value$(TB).csv");
                g = read_data("SavingOfComputation/DataOfCell_$cell/AnalysisSolverData/1_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/Solv_Gnorm$(TB).csv");

                if TB == type_basis[2] 
                    ScD[nbasK-nbreBasis[1]+1] = s[end]
                else
                    Sc[nbasK-nbreBasis[1]+1,nbasH-nbreBasis[1]+1] = s[end]
                    ScG[nbasK-nbreBasis[1]+1,nbasH-nbreBasis[1]+1] = g[end]
                    Scit[nbasK-nbreBasis[1]+1,nbasH-nbreBasis[1]+1] = length(g)
                end
                if nbasK == nbreBasis[end] && TB == type_basis[1]# make HeatMap
                    folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","HeatMap"], "", "", 1)
                    backend(:plotly)
                    X = nbreBasis
                    Y = nbreBasis
                    P = plot(heatmap(X, Y, Sc, color=:buda))
                    xlabel!("Number of k basis")
                    ylabel!("Number of h basis")
                    savefig(P, joinpath(folder_path, "GLMscore_$(fct_fit).html"))

                    P = plot(heatmap(X, Y, ScG, color=:buda))
                    xlabel!("Number of k basis")
                    ylabel!("Number of h basis")
                    savefig(P, joinpath(folder_path, "GLMsgrad_$(fct_fit).html"))

                    P = plot(heatmap(X, Y, Scit, color=:buda))
                    xlabel!("Number of k basis")
                    ylabel!("Number of h basis")
                    savefig(P, joinpath(folder_path, "GLMnumofiteration_$(fct_fit).html"))

                end
                if nbasK == nbreBasis[end] && TB == type_basis[2] # Make a plot of score to compare article's basis and Denis' basis
                    folder_path = save_data(["SavingOfComputation","Figure","Cell_$(cell)","ComparisonBasisDesign"], "", "", 1)
                    gr()
                    P = plot(nbreBasis, Sc[:,5], label="Article Basis" )
                    plot!(nbreBasis, ScD, label="Own Basis")
                    xlabel!("Number of k basis")
                    ylabel!("Score")
                    savefig(P, joinpath(folder_path, "FigureComp_of_GLMscore_for_cell$(cell).html"))

                end
            end

        end
    end
end

if Plot_best_filters
    nbasK = 6
    nbasH = 7
    n_tP = 1
    fct_fit = LLAICf[1]
    TB = ""
    dt = 0.1
    for cell in cell_num
        kbasis = read_data("SavingOfComputation/DataOfCell_$(cell)/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/kbasis$(TB).csv")
        hbasis = read_data("SavingOfComputation/DataOfCell_$(cell)/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/hbasis$(TB).csv")
        prs = read_data("SavingOfComputation/DataOfCell_$(cell)/$(n_tP)_Training_Period/$(nbasK)_basisK/$(nbasH)_basisH/$(fct_fit)/prs$(TB).csv")
        k = kbasis*prs[1:nbasK]
        h = hbasis*prs[nbasK+1:end-1]
        Pk = plot(-length(k)*dt:dt:-dt,k,label="")
        plot!(xlabel=L"Time~[ms]",xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12)
        Ph = plot(-length(h)*dt:dt:-dt,reverse(h),label="")
        plot!(xlabel=L"Time~[ms]",xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12)
        folder_path = save_data(["SavingOfComputation","Figures","Cell$(cell)"], "", "", 1)
        savefig(Pk, joinpath(folder_path,"k.pdf"))
        savefig(Ph, joinpath(folder_path,"h.pdf"))
    end

end

println("\007")