


using Plots, BenchmarkTools, GLM, Optim, Interpolations, DSP, FFTW, Base, LinearAlgebra, Statistics
using DelimitedFiles
using CSV, DataFrames
using DynamicAxisWarping: dtw, fastdtw
using DifferentialEquations, LaTeXStrings
# using Base
using Logging
Logging.disable_logging(Logging.Warn)



function all_parameters(cellType)
    #       a           b       c     d          I         dt
    pars = [0.02      0.2     -65     6          14       0.1 ;      # 1. tonic spiking
            0.02      0.25    -65     6          .50      0.1 ;      # 2. phasic spiking
            0.02      0.2     -50     2          10       0.1 ;      # 3. tonic bursting
            0.02      0.25    -55     0.05       .6       0.1 ;      # 4. phasic bursting
            0.02      0.2     -55     4          10       0.1 ;      # 5. mixed mode
            0.01      0.2     -65     5          20       0.1 ;      # 6. spike frequency adaptation
            0.02      -0.1    -55     6          25       0.1 ;      # 7. Class 1
            0.2       0.26    -65     0          .5       0.1 ;      # 8. Class 2
            0.02      0.2     -65     6          3.49     0.1 ;      # 9. spike latency
            0.05      0.26    -60     0          0        1   ;      # 10. subthreshold oscillations
            0.1       0.26    -60     -1         .3       0.5 ;      # 11. resonator
            0.02      -0.1    -55     6          27.4     0.5 ;      # 12. integrator
            0.03      0.25    -60     4          -5       0.1 ;      # 13. rebound spike
            0.03      0.25    -52     0          -5       0.1 ;      # 14. rebound burst
            0.03      0.25    -60     4          2.3      1   ;      # 15. threshold variability
            1         1.5     -60     0          26.1     0.05;      # 16. bistability
            1         0.2     -60     -21        0        0.1 ;      # 17. depolarizing after-potential
            0.02      1       -55     4          20       0.1 ;      # 18. accomodation
            -0.02     -1      -60     8          70       0.1 ;      # 19. inhibition-induced spiking
            -0.026    -1      -45     0          70       0.1 ;      # 20. inhibition-induced bursting
            1         1.5     -60     0          26.1     0.05 ]     # 21. bistability 2 (Not in original Izhikevich paper)
    
    return pars[cellType,:]
end

function I_app_fct(cellType, T)
    TITLES = [  "Tonic spiking",
                "Phasic spiking",
                "Tonic bursting",
                "Phasic bursting",
                "Mixed mode",
                "Spike frequency adaptation",
                "Class 1",
                "Class 2",
                "Spike latency",
                "Subthreshold oscillations",
                "Resonator",
                "Integrator",
                "Rebound spike",
                "Rebound burst",
                "Threshold variability",
                "Bistability 1",
                "Depolarizing after-potential",
                "Accomodation",
                "Inhibition-induced spiking",
                "Inhibition-induced bursting",
                "Bistability 2"  ]
        title = TITLES[cellType]

    if cellType in [10,17]
        println("This is a \"", title ,"\" behavior, which can't be captured by a GLM.  No example stimulus has been designed for this cell type.")
        dt = all_parameters(cellType)[6]
        t = dt:dt:T
        I = zeros(length(t),1)
        return I,title,T,t
    end
    if isempty(T) == true
        T = 1000
    end

    Ival = all_parameters(cellType)[5]
    dt = all_parameters(cellType)[6]
    dt = round(dt, digits=2)
    t = round.(dt:dt:T, digits=2)
    I = zeros(length(t),1)
    stepLength = 500
    nStepsUp = floor(T/stepLength/2)

    if cellType in [1,2,3,4,5,6,10,19,20]
        if cellType in [19,20]
            I = 80*ones(length(t),1)
        end
        for i in 1:nStepsUp
            idx = t.>  (stepLength+stepLength*2*(i-1)) .&& t.<  (stepLength*2*(i)+1)
            I[idx] .= Ival;
        end
    elseif cellType in [7,8]
        if cellType == 7
            stepSizes = 15:1:30
        elseif cellType ==8
            stepSizes = .1:.025:.7
        end
        for i in 1:length(stepSizes)
            idx = t.>(stepLength+stepLength*2*(i-1)) .&& t.<(stepLength*2*(i)+1);
            I[idx] .= stepSizes[i];
        end
    elseif cellType in [9]
        stepLength = 150
        nStepsUp = floor(T/stepLength/2)
        for i in 1:nStepsUp
            idx = t.>(stepLength*1.94+stepLength*2*(i-1)) .&& t.<(stepLength*2*(i)+1)
            I[idx] .= Ival
        end
    elseif cellType in [11]
        stepLength = 150
        nStepsUp = floor(T/stepLength/2)
        for i in 2:nStepsUp       
            pulseLength = round(5/dt);
            idx = t.>(stepLength+stepLength*2*(i-1)) .&& t.<(stepLength+stepLength*2*(i-1)+pulseLength)
            I[idx] .= Ival;
         # second pulse
            idx = t.>(stepLength+stepLength*2*(i-1)+pulseLength+2*i+pulseLength/2) .&& t.<(stepLength+stepLength*2*(i-1)+2*pulseLength+2*i+pulseLength/2);
            I[idx] .= Ival;
        end
    elseif cellType in [12]
        stepLength = 250
        nStepsUp = floor(T/stepLength/2)
        for i in 3:nStepsUp
            pulseLength = round(4/dt)
            idx = t.>(stepLength+stepLength*2*(i-1)) .&& t.<(stepLength+stepLength*2*(i-1)+pulseLength);
            I[idx] .= Ival;
         # second pulse
            idx = t.>(stepLength+stepLength*2*(i-1)+pulseLength+6*i+pulseLength/2) .&& t.<(stepLength+stepLength*2*(i-1)+2*pulseLength+6*i+pulseLength/2);
            I[idx] .= Ival;
        end
    elseif cellType in [13 14]
        for i in 1:nStepsUp
            idx = t.>(stepLength*1.6+stepLength*2*(i-1)) .&& t.<(stepLength*2*(i)+1)
            I[idx] .= Ival
        end
    elseif cellType in [15]
        dur = 1/dt
        for i in 1:nStepsUp*2
            idx_f = stepLength*i-dur :1: stepLength*i
            L = length(idx_f)
            idx = Int[]
            for i in 1:L
                push!(idx,convert(Int,idx_f[i]))
            end 
            I[idx] .= Ival
            if mod(i,2) == 1
                I[idx.-25] .= -Ival
            end
        end
    elseif cellType in [16 21] 
        if cellType == 16
            pulsePolarity = 1
        elseif cellType == 21
            pulsePolarity = -1
        end
        stepLength = 50
        nStepsUp = floor(T/stepLength)
        I = I.-65
        pulseDir = 2
        delay = -3
        for i in 1:nStepsUp
            if mod(i,2) == 1
                idx = t.>(stepLength+stepLength*(i-1)) .&& t.<(stepLength+stepLength*(i-1)+pulseDir);
                I[idx] .= I[idx].+Ival
            else
                idx = t.>(delay+stepLength+stepLength*(i-1)) .&& t.<(delay+stepLength+stepLength*(i-1)+pulseDir);
                I[idx] .= I[idx].+(Ival*pulsePolarity)
            end
        end
    elseif cellType in [18] 
        baseline = -70;
        I = baseline*ones(size(I))
        for i = 1:nStepsUp
            if mod(i,2) == 1
                idx = t.>(stepLength+stepLength*2*(i-1)) .&& t.<(stepLength*2*(i)+1)
                I[idx] .= LinRange(baseline,baseline+Ival,sum(idx))
            else
                idx = t.>(stepLength*1.9+stepLength*2*(i-1)) .&& t.<(stepLength*2*(i)+1)
                I[idx] .= LinRange(baseline,baseline+Ival,sum(idx))
                
            end
        end   
    end
    I = I[1:length(t)]
    return I,title,T,t
end

function generate_stimulation(cellType)
    ~,~,~,~,~,dt = all_parameters(cellType)
    T = 10000
    if cellType in [7,8]
        T = 20000
    end
    I, title, T, time = I_app_fct(cellType,T)
    return I, dt, T, title, time
end

function simulate_izhikevich(cellType, I, dt)
    
    a,b,c,d,~,~ = all_parameters(cellType)
    
    T = length(I)*dt
    t = dt:dt:T
    
    ## initialize variables
    threshold = 30;
    v = Vector{Float64}(undef, length(t))
    u = Vector{Float64}(undef, length(t))
    spikes = zeros(length(t),1);
    
    # different initial v and u values to start different neuron types near
    # stable fixed point (prevent spiking in absence of inputs near t=0)
    if cellType in [16 21] # if bistable
        v[1] = -54;
        u[1] = -77; 
    elseif cellType == 12 # integrator
        v[1] = -90;
        u[1] = 0;
    elseif cellType in [19 20] # inhibition-induced spiking/bursting
        v[1] = -100;
        u[1] = 80;
    else
        v[1] = -70; 
        u[1] = -14; 
    end
    
    # Izhikevich model doesn't show this kind of bistability, so simulate
    # responses using first form of bistability
    Iplot = I; # trick
    if cellType == 21
        for i in 1:length(I)
            I[i] = abs(I[i]+65)-65;
        end
    end
    
    ## run model
    for tt = 1:length(I)-1
        dvdt = 0.04*v[tt]^2 + 5*v[tt] +140 - u[tt] + I[tt];
        v[tt+1] = v[tt] + dvdt*dt;
        dudt = a*(b*v[tt+1]-u[tt]);
        u[tt+1] = u[tt] + dudt*dt;
        if v[tt+1]>threshold
            v[tt] = threshold;  # makes spikes of uniform height
            v[tt+1] = c;
            u[tt+1] = u[tt+1] + d;
            spikes[tt+1] = 1;
        end
    end    
    return v, u, spikes
end

function MIN_or_MAX_matrix_numer(b,A,operation)
    """
    operation = 1 is max
    operation = -1 is min
    """

    if operation == -1
        for i in 1:size(A)[1]
            for j in 1:size(A)[2]
                A[i,j] = min(A[i,j],b)
            end
        end
    else
        for i in 1:size(A)[1]
            for j in 1:size(A)[2]
                A[i,j] = max(A[i,j],b)
            end
        end
    end
    return A
end

function ff(x, c, dc) 
    Z = (x .- c) .* π ./ dc ./ 2
    A = MIN_or_MAX_matrix_numer(π, Z, -1) 
    B = MIN_or_MAX_matrix_numer(-π, A, 1)
    C = (cos.(B) .+ 1) ./ 2
    return C  # Raised cosine basis vector
end

function makeBasis_PostSpike(ihprs,dt,iht0=[])

    b = ihprs[:b]
    hpeaks = ihprs[:hpeaks]
    absref = ihprs[:absref]
    ncols = ihprs[:ncols]

    if absref >= dt
        ncols = ncols - 1
    elseif absref > 0
        error("Error with absref")
    end
    
    nlin(x) = log.(exp(1),x .+ 10^(-20))
    invnl(x) = exp(x) - 10^(-20)  # Inverse nonlinearity

    # Generate basis of raised cosines
    yrnge = nlin(hpeaks .+ b) 
    db = (yrnge[2] - yrnge[1]) / (ncols - 1)  # Spacing between raised cosine peaks
    ctrs = yrnge[1]:db:yrnge[2]  # Centers for basis vectors
    mxt = invnl(yrnge[2] + 2 * db) - b  # Maximum time bin
    iht = (0:dt:mxt)';
    nt = length(iht)  # Number of points in kt0

    ihbasis = ff(repeat(nlin(iht .+ b)', 1, ncols), repeat(ctrs', nt, 1), db)
    
    ii = findall(iht .<= hpeaks[1]);
    ihbasis[ii] .= 1;

    if absref >= dt
        ii = findall(iht .< absref);
        ih0 = zeros(size(ihbasis)[1],1)';
        ih0[ii] .= 1;
        ih0 = ih0'
        L = length(ii)
        for indice in 1:L
            II = ii[indice][2]
            for indice_bis in 1:size(ihbasis)[2]
                ihbasis[II,indice_bis] = 0;
            end
        end
        ihbasis = hcat(ih0,ihbasis);
    end
    ihbas, ~ = qr(ihbasis);  
            
    return iht, ihbas, ihbasis    
end
    
function negloglike_glm_basis(prs, NL::Function, xconvki, yconvhi, y, dt, refreshRate, fct_fit)

    nkbasis = size(xconvki)[2]; # number of basis functions for k

    kprs = prs[1:nkbasis]; # k basis functions weighted by given parameters
    hprs = prs[nkbasis+1:end-1]; #  basis functions weighted by given parameters
    dc = prs[end]; # dc current (accounts for mean spike rate)

    xconvk_dc = xconvki*kprs .+ dc;    
    yconvh = yconvhi*hprs;
    
    g = (xconvk_dc + yconvh) # ./(length(y)*dt)
    lambda = NL(g)

    negloglike = (-y'*g)[1] + dt*sum(lambda)/refreshRate  # negative log likelihood 

    if fct_fit == "AICfit"
        negloglike = 2* (length(prs) - (-1*negloglike))
    end
    return [negloglike]
end

function fft_padded(s, n)
    signal = zeros(n,1)
    lg = minimum([length(s), n])
    for i in 1:lg
        signal[i] = s[i]
    end
    return fft(signal)
end
    
function sameconv(A, B) 
    # Because I dont found the convolution function in Julia
    am = size(A)[1]
    bm = size(B)[1]

    if length(size(A)) == 1
        an = 1
    else 
        an = size(A)[2]
    end
    if length(size(B)) == 1
        bn = 1
    else 
        bn = size(B)[2]
    end
    nn = am+bm-1
    
    a = fft_padded(A,nn)
    b = fft_padded(reverse(B),nn)
    G = ifft(a .* b)
    G = G[1:am,:]
    G = real(G[1:am,:])
    return G
end

function Post_Basis_Denis(L)
    B_exp = zeros(L,1);
    B_thin = zeros(L,1)
    B_3 = zeros(L,1)
    B_4 = zeros(L,1)
    B_5 = zeros(L,1)
    B_6 = zeros(L,1)
    B_sigmo = zeros(L,1);
    expo(i,L) = exp(-i/(L/100))
    for i in 1:L
        B_exp[i] = expo(i,L)^3
        B_thin[i] =  (-expo(i/1.25,L)*(expo(i/1.25,L)-1))^1
        B_3[i] =  (-expo(i/5,L)*(expo(i/5,L)-1))^1
        B_4[i] =  (-expo(i/10,L)*(expo(i/10,L)-1))^4
        B_5[i] =  (-expo(i/20,L)*(expo(i/20,L)-1))^4
        B_6[i] =  (-expo(i/40,L)*(expo(i/40,L)-1))^4
        z = ((i+L/2)/L) * (i-2*L/4)/L
        if z > 0
            z = 0
        end
        B_sigmo[i] = maximum(B_sigmo) + z^2
    end
    B_exp[:] =  B_exp[:]/maximum(B_exp)
    B_thin[:] =  B_thin[:]/maximum(B_thin)
    B_3[:] =  B_3[:]/maximum(B_3)
    B_4[:] =  B_4[:]/maximum(B_4)
    B_5[:] =  B_5[:]/maximum(B_5)
    B_6[:] =  B_6[:]/maximum(B_6)
    i = L
    while true
        if B_thin[i] < maximum(B_thin)
            B_thin[i] = B_thin[i]^2
        else
            break
        end
        i = i -1
    end
    i = L
    while true
        if B_3[i] < maximum(B_3)
            B_3[i] = B_3[i]^4
        else
            break
        end
        i = i-1
    end
    i = 1
    while true
        if B_4[i] < maximum(B_4)
            B_4[i] = B_4[i]^2
        else
            break
        end
        i = i +1
    end
    i = 1
    while true
        if B_5[i] < maximum(B_5)
            B_5[i] = B_5[i]^2
        else
            break
        end
        i = i +1
    end
    i = 1
    while true    
        if B_6[i] < maximum(B_6)
            B_6[i] = B_6[i]^2
        else
            break
        end
        i = i +1
    end

    B = (B_sigmo[:].-minimum(B_sigmo))
    B_sigmo[:] =  (B/maximum(B)).^20;

    ihbasis = hcat(B_exp,B_thin,B_3,B_4,B_5,B_6,B_sigmo)

    return ihbasis
end
    
function fit_glm(x, y, dt, nkt, kbasprs, ihbasprs, prs, maxIter, tolFun, solver, type_basis, fct_fit)

    refreshRate = 1000 / dt # Calculate the refresh rate (stimulus in ms, sampled at dt)
    kbasisTemp = makeBasis_StimKernel(kbasprs, nkt) # Create stimulus kernel basis functions
    kbasisTemp = kbasisTemp[1]
    nkb = size(kbasisTemp)[2]
    lenkb = size(kbasisTemp)[1]
    kbasis = zeros(convert(Int, lenkb/dt), nkb)
  
    for bNum in 1 : nkb
        itp = LinearInterpolation(1:lenkb, kbasisTemp[:, bNum])
        kbasis[:, bNum]  = itp(LinRange(1, lenkb, convert(Int, lenkb/dt)))
    end
   
    ht, hbas, hbasis = makeBasis_PostSpike(ihbasprs, dt) # Create post-spike basis functions
    hbasis = vcat(zeros(1, ihbasprs[:ncols]), hbasis)  # enforce causality

    if type_basis == "Denis"
        L = 2760
        hbasis = Post_Basis_Denis(L)
        hbasis = vcat(zeros(1, size(hbasis)[2]), hbasis)
    end 
    nkbasis = size(kbasis)[2]
    nhbasis = size(hbasis)[2]
    if isempty(prs)
        prs = zeros(nkbasis + nhbasis + 1)
    end

    xconvki = zeros(size(y, 1), nkbasis)
    yconvhi = zeros(size(y, 1), nhbasis)

    for knum in 1:nkbasis
        xconvki[:, knum] = sameconv(x, kbasis[:, knum])
    end

    for hnum in 1:nhbasis
        yconvhi[:, hnum] = sameconv(y, reverse(hbasis[:, hnum]))
    end

    opts = Optim.Options(
        show_trace = true,
        store_trace = true,
        iterations = maxIter,
        show_every = 200,
        allow_f_increases = true,
        x_tol = tolFun,
        f_tol = tolFun,
    )

    fct(x) = exp.(x)
    fneglogli(prs) = negloglike_glm_basis(prs, fct, xconvki, yconvhi, y, dt, refreshRate, fct_fit)
    result = optimize(prs -> fneglogli(prs)[1], prs, solver, opts, autodiff=:forward)   # time-consumming

    prs = Optim.minimizer(result)
    k = kbasis * prs[1:nkbasis]
    h = hbasis * prs[nkbasis+1:end-1]
    dc = prs[end]

    trace = Optim.trace(result)

    return k, h, dc, prs, kbasis, hbasis, trace
end

function normalizecols(X)
    # Normalize columns of a matrix to have unit L2 norm
    return X ./ sqrt.(sum(X.^2, dims=1))
end

function makeBasis_StimKernel(kbasprs, nkt)
    neye = kbasprs[:neye]
    ncos = kbasprs[:ncos]
    kpeaks = kbasprs[:kpeaks]
    b = kbasprs[:b]
    kdt = 1  # Spacing of x-axis, must be in units of 1

    # Nonlinearity for stretching x-axis (and its inverse)
    nlin(x) = log.(exp(1),x .+ 10^(-20)) # L10
    invnl(x) = exp(x) - 10^(-20)  # Inverse nonlinearity

    # Generate basis of raised cosines
    yrnge = nlin(kpeaks .+ b) 
    db = (yrnge[2] - yrnge[1]) / (ncos - 1)  # Spacing between raised cosine peaks
    ctrs = yrnge[1]:db:yrnge[2]  # Centers for basis vectors
    mxt = invnl(yrnge[2] + 2 * db) - b  # Maximum time bin
    kt0 = 0:kdt:mxt
    nt = length(kt0)  # Number of points in kt0
    
    kbasis0 = ff(repeat(nlin(kt0 .+ b), 1, ncos), repeat(ctrs', nt, 1), db)
    # Concatenate identity-vectors
    nkt0 = size(kt0, 1)
    kbasis = [[I(neye); zeros(nkt0, neye)] vcat(zeros(neye, ncos), kbasis0)]
    kbasis = reverse(kbasis, dims=1)  # Flip so fine timescales are at the end
    nkt0 = size(kbasis, 1)

    if nkt0 < nkt
        # Padding basis with zeros
        kbasis = [zeros(nkt - nkt0, ncos + neye); kbasis]
    elseif nkt0 > nkt
        # Removing rows from basis
        kbasis = kbasis[end - nkt + 1:end, :]
    end

    # Normalize columns
    kbasis = normalizecols(kbasis)
    kbas = copy(kbasis)  # You can return kbas or kbasis depending on your needs
    return kbas, kbasis
end

function simulate_glm(x,dt,k,h,dc,runs)
    if isempty(runs)
        runs = 5;
    end

    ## generate data with fitted GLM
    nTimePts = length(x);
    refreshRate = 1000/dt; # stimulus in ms, sampled at dt

    NL(x) = exp(x);
    g = zeros(nTimePts+length(h),runs); # filtered stimulus + dc
    y = zeros(nTimePts,runs); # initialize response vector (pad with zeros in order to convolve with post-spike filter)
    r = zeros(nTimePts+length(h)-1,runs); # firing rate (output of nonlinearity)
    hcurr = zeros(size(g));
    stimcurr = sameconv(x,k);
    Iinj = stimcurr .+ dc;
    
    for runNum in 1:runs
        g[:,runNum] = [Iinj; zeros(length(h),1)]; # injected current includes DC drive
        ### loop to get responses, incorporate post-spike filter
        for t in 1:nTimePts
            r[t,runNum] = NL(g[t,runNum]);  # firing rate
            if rand() < (1-exp(-r[t,runNum]/refreshRate)); # 1-P(0 spikes)
                y[t,runNum] = 1;
                g[t:t+length(h)-1,runNum] = g[t:t+length(h)-1,runNum] + h;  # add post-spike filter
                hcurr[t:t+length(h)-1,runNum] = hcurr[t:t+length(h)-1,runNum] + h;
            end
        end
    end
    hcurr = hcurr[1:nTimePts,:];  # trim zero padding
    r = r[1:nTimePts,:];  # trim zero padding
    return [y, stimcurr, hcurr, r] 
end

function access_and_create_folder(folder_path)
    # Verify is the folder exists
    lock(lk_access_and_create_folder) do
        if isdir(folder_path) == false
            mkdir(folder_path)
        end
    end
end

function save_data(folder, file, data, tricktocreate=0)
    path = folder[1]
    access_and_create_folder(path)
    for l in 2:1:length(folder)
        path = joinpath(path,folder[l])
        access_and_create_folder(path)
    end
    if tricktocreate==0
        path = joinpath(path,file)
        nbre_silent = sum(data.==0)
        nbre_spikes = sum(data.==1)
        if nbre_silent!= length(data) && nbre_spikes!= length(data) && (nbre_silent+nbre_spikes)==length(data) # is spike
            data_to_save = zeros(nbre_spikes+1) # the last will be to length of the vector
            data_to_save[end] = length(data)
            j = 1
            for i in 1:length(data)
                if data[i] == 1
                    data_to_save[j] = i
                    j += 1
                end
            end
            writedlm(path, data_to_save, ',')
        else
            writedlm(path, data, ',')
        end
    else
        return path
    end
end

function read_data(path, is_spike_data=0)
    data = CSV.File(path, header=false) #|> DataFrame
    size_data = size(data)

    if length(data[1]) == 1 # not second dimension
        col_max = 1
    else
        col_max = length(data[1])
    end

    V = zeros(length(data), col_max)
    for c in 1:1:length(data)
        for i in 1:1:col_max
            if typeof(data[c][i]) == typeof("data")
                A = data[c][i]
                global condbis = false
                for bis in 2:length(data[c][i])
                    if A[bis] == '.'
                        global condbis = true
                    end
                    if A[bis] == '-' || A[bis] == '+'  || (A[bis] == '.' && condbis)
                        A = A[1:bis-1]
                        break
                    end
                end
                V[c,i] = parse(Float16,A)
            else
                V[c,i] = data[c][i]
            end
        end
    end
    if is_spike_data==1
        V = convert.(Int,V)
        sp = zeros(V[end])
        j = 1
        for i in 1:length(sp)-1
            if V[j] == i
                sp[i] = 1
                j +=1
            end
        end
        V = sp
    end
    return V
end

function save_title(path,title)
    open(path, "w") do file
    write(file, title)
    end
end
