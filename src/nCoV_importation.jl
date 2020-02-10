# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

# # 2019-nCoV outbreak analysis
#
# ## Background
# A novel coronavirus (2019-nCoV) outbreak has been continuing in China, while a few imported cases are observed in the neighbouring countries. Although the spread across China suggests a certain amount of human-to-human transmisison, there has not been any observed in those countries which saw travel cases. This suggests that not all symptomatic cases lead to a secondary transmission, which was also the case in the past SARS/MERS outbreaks. Furthermore, even if any subclinical cases had been imported into these countries undetected, at least such cases did not contributed to secondary transmissions from which a detectable case originates.
#

# ## Model
# Let $J_t$ and $I_t$ be the incidence of imported and local cases of nCoV detected in a country outside China.
# Let $q_t$ be the (potentially time-dependent) detection probability of nCoV infection, which is understood as the combination of both symptomatic ratio and proper reporting, and $s_\tau$ be the distribution of the serial interval (i.e. time between the linked infection pairs) of nCoV.
#
# Assuming that the risk of importing an nCoV case follows an exponential hazard function, reflecting an ongoing outbreak in China as of February 2020, the overall (including those undetected) number of imported cases $j_t$ is given as 
# $$
# j_t\sim \mathrm{Pois}(\lambda_t), \\
# h_t=h_0\exp(r t)
# $$
#
# We model the offspring distribution of nCoV by a negative binomial distribution $\mathrm{NB}\left(k,\frac{k}{R_0+k}\right)$, where $R$ is the mean (i.e. the basic reproduction number) and $k$ is the overdispersion parameter. The reproductive property of the negative binomial distriubution assures that the number of secondary infections caused by the overall infector at time $t$ follows $\mathrm{NB}\left(k(i_t+j_t),\frac{k}{R_0+k}\right)$, and we assume these offsprings are distributed on the timeline $t$ according to the serial interval distribution $s_\tau$.
#
# Since sampling from a negative binomial distribution is identical to sequentially sampling from Gamma and Poisson distributions, we can construct the renewal process as
# $$
# i_t \sim \mathrm{Pois}\left(\sum_{\tau=1}^\infty \lambda_{t-\tau},s_\tau \right),\\
# \lambda_t\sim\mathrm{Gamma}\left(k(i_{t-\tau}+j_{t-\tau}),\frac{R_0}{k}\right).
# $$
# Here, $\lambda_t$ is the total force of infection caused by $i_t$: infectious individuals with onset $t$.
#
# The observation of cases is assumed to follow the binomial sampling:
# $$
# I_t\sim \mathrm{Binom}(i_t,q_t), \\
# J_t\sim \mathrm{Binom}(j_t,q_t)
# $$
#
# We assumed the probability of detection $q_t$ might have been lower before the epidemic became widely recognised. Because of reporting delays, the most recent data might have also been underreported. We model $q_t$ as
# $$
# q_t=Q(t)\delta(t), \\
# \delta(t)=1-\exp\left(\frac{1}{d}(T-t)\right),
# $$
# where $Q(t)$ is the baseline reporting probability at time $t$ and $\delta(t)$ is the factor reflecting the delay. We assume the distribution of reporting delays follow an exponential distribution with mean $d$.

# ## Statistical analysis
# We assumed the observed imported/local cases $I_t,J_t$ and the serial interval distribution $S_t$ are given. Of the unkown variables, $i_t$, $j_t$ and $\lambda_t$ are sampled by the particle-Gibbs algorithm and the remaining variables $R_0, k, h_0,r,d$ are sampled by No-U-turn sampler (NUTS).

#



# ## Data
# Reported cases outsie China by onset dates were extracted from the WHO situation reports (Situation Report 14 as of 4/2/2020, accessed on the same day). We considered cases with a travel history in china as imported cases, and cases labelled as "locally acquired" as local cases. We did not included those labelled as "Under investigation" in this analysis.

# Show incidence data
using DataFrames, Dates, PyPlot
# Imported and local cases outside China
# Source (accessed 4/2/2020): https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200204-sitrep-15-ncov.pdf
dates=Date("2019-12-31"):Day(1):Date("2020-2-2")
china_hubei  =[1,0,0,2,0,1,0,0,0,1,1,0,0,1,2,0,3,3,1,3,3,7,3,9,6,8,5,4,3,3,4,1,0,0]
china_unknown=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
localcases   =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,2,0,0,0,1,1,0,0,1,2,0,0,0,0]
u_inv        =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1,0,1,0]
casedata=DataFrame([dates,china_hubei,china_unknown,localcases,u_inv], [:date,:china_hubei,:china_unknown,:loc,:underinv])
barplots=PyPlot.bar.(Ref(1:length(casedata.date)),[Vector(casedata[:,c]) for c in 2:5],0.8,[cumsum(hcat(zeros(Int,length(casedata.date)),Matrix(casedata[:,2:5])),dims=2)[:,c] for c in 1:4])
PyPlot.xticks(1:length(casedata.date),dates,rotation="vertical")
PyPlot.legend(getindex.(barplots,1),["China (Hubei)","China(Location unknown)","Locally acquired","Under investigation"]);

# Data preparation
const initdate=Date("2019-9-13")
const newyeardate=Date("2020-1-1")
# Possible (earliest bound of) the estimted start of the outbreak
# Source: http://virological.org/t/preliminary-phylogenetic-analysis-of-11-ncov2019-genomes-2020-01-19/329
const timelines=initdate:Day(1):Date("2020-2-3")
const tlen=length(timelines)
imported=zeros(Int,tlen)
loc=zeros(Int,tlen)
datesid=findall(x->x in dates,timelines)
imported[datesid].=casedata.china_hubei.+casedata.china_unknown
loc[datesid].=casedata.loc
observed=(imported=imported,loc=loc);

# ## Setups

# +
# Packages
using Mamba, Distributions, LinearAlgebra
# Distributions
NBmu(mu,k)=NegativeBinomial(k,mu/(mu+k))
Gmusd(mu,sd)=Gamma(mu^2/sd^2, sd^2/mu)
module DSI
using Distributions
struct DiscreteSerialInterval{T<:NamedTuple,F<:AbstractFloat, D<:Distribution}
    params::T; dist::D; distvec::Vector{F}
end
end
function DSIconstruct(distconstructor,params,tlen)
    dist=distconstructor(params...)
    distvec=cdf.(dist,0:tlen) |> diff
    DSI.DiscreteSerialInterval(params,dist,distvec)
end
SIparams=(μ=7.0,σ=3.0)
SIdist=Gmusd(SIparams...)
serialint=DSIconstruct(Gamma,NamedTuple{(:α,:θ)}(params(SIdist)),tlen)

# utils
function sysresample(weights)
    size=length(weights)
    weights=weights/sum(weights) # normalise
    cumweights=cumsum(weights)
    randomiser=floor.(Int,cumweights.*size.+rand())
    freqs=randomiser.-[0; randomiser[1:end-1]]
    return(reduce(vcat,fill.(1:size,freqs)))
end

# +
# ## Simulation functions
using Distributions, StatsFuns, StatsBase, SpecialFunctions, Parameters
function particlegibbs!(infness_pts::NamedTuple{(:imported,:loc),T} where T,cases_pts::NamedTuple{(:imported,:loc),T} where T,hazard_pts::NamedTuple{(:imported,:loc),T} where T,nbparm,detectprob,gtimevec,observed)
    nsample=size(infness_pts.imported,2)
    tlen=length(observed.loc)
    lweights=zeros(nsample)
    llkh=0.0
    α,p=nbparm
    θ=(1-p)/p
    for t in 1:length(observed.loc)
        # draw cases
        @views cases_pts.imported[t,2:end].=observed.imported[t].+rand.(Poisson.((1-detectprob[t]).*hazard_pts.imported[t]))
        @views cases_pts.loc[t,2:end].=observed.loc[t].+rand.(Poisson.((1-detectprob[t]).*hazard_pts.loc[t,2:end]))
        for tag in keys(cases_pts)
            #draw gamma: total offsprings
            nonzerocase=cases_pts[tag][t,2:end].!=0 # to avoid Gamma(0,θ)
            if sum(nonzerocase)!=0 @views infness_pts[tag][t,2:end][nonzerocase].=rand.(Gamma.(α.*cases_pts[tag][t,2:end][nonzerocase],θ)) end
            #distribute infness_pts on timeline
            hazard_pts.loc[t+1:end,2:end] .+= (@view infness_pts[tag][t:t,2:end]).*gtimevec[1:length(observed.loc)-t]
            @views lweights.+=logpdf.(Poisson.(detectprob[t].*hazard_pts[tag][t,:]),observed[tag][t])
        end
        if all(lweights.≤-Inf) llkh=-Inf;break end
        if 2logsumexp(lweights)-logsumexp(2 .*lweights)< log(nsample)-log(2) || t==length(observed.loc)
            llkh+=logsumexp(lweights)-log(nsample)
            lweights.-=maximum(lweights)
            newid=sysresample(exp.(lweights))
            @views newid.=[1;newid[Not(rand(1:nsample))]]
            for tag in keys(cases_pts)
                @views infness_pts[tag][1:t,:].=infness_pts[tag][1:t,newid]
                @views cases_pts[tag][1:t,:].=cases_pts[tag][1:t,newid]
            end
            @views hazard_pts.loc[1:t,:].=hazard_pts.loc[1:t,newid]
            lweights.=0.0
        end
        # resample
        
    end
    return(llkh)
end
function infcasesgibbs!(paths,nsamples,branchdist,gtimedist,observed,detectprob,tlen)
    particles=(infness_pts=(imported=zeros(tlen,nsample), loc=zeros(tlen,nsample)),
    hazard_pts=(imported=paths.hazard.imported,loc=zeros(tlen,nsample)),
    cases_pts=(imported=zeros(Int,tlen,nsample),loc=zeros(Int,tlen,nsample)))
    @unpack infness_pts,cases_pts,hazard_pts = particles
    @unpack infness,cases,hazard=paths

    # Pass the reserved particle for conditional particle filter
    for tag in keys(infness)
        infness_pts[tag][:,1].=infness[tag]
        cases_pts[tag][:,1].=cases[tag]
    end
    hazard_pts.loc[:,1].=hazard.loc
    gtimevec=diff(cdf.(gtimedist,0:tlen))
    counts=0
    ll=0.0
    while true
        infness_pts.imported.=0.0
        infness_pts.loc.=0.0
        hazard_pts.loc.=0.0
        ll=particlegibbs!(infness_pts,cases_pts,hazard_pts,params(branchdist),detectprob,gtimevec,observed)
        counts+=1
        if ll>-Inf break end
        if counts>100 error("infness could not be sampled in 100 SMC iterations") end
    end

    # sample one particle
    sampleid=sample(1:nsample)
    if isnan(sampleid) sanpleid=1 end
    @views infness.imported.=infness_pts.imported[:,sampleid]
    @views infness.loc.=infness_pts.loc[:,sampleid]
    @views cases.imported.=cases_pts.imported[:,sampleid]
    @views cases.loc.=cases_pts.loc[:,sampleid]
    @views hazard.loc.=hazard_pts.loc[:,sampleid]
    ll
end

function llnbdist(nbparm,paths)
    @unpack infness,cases=paths
    α,p=nbparm
    θ=(1-p)/p
    ll=0.0
    tlen=length(infness.imported)
    for tag in keys(infness)
        for t in 1:tlen if (cases[tag][t]==0 && infness[tag][t]>0.0) ll=-Inf; break end end
        ll+=sum((logpdf(Gamma(α*cases[tag][t],θ),infness[tag][t]) for t in 1:tlen if cases[tag][t]>0))
        if isnan(ll) ll=-Inf;break end
    end
    ll
end
function lldetectprob(detectprob,paths,observed)
    @unpack cases,infness=paths
    ll=0.0
    tlen=length(infness.imported)
    for tag in keys(infness)
        ll+=sum((logpdf(Binomial(Int(cases[tag][t]),detectprob[t]),observed[tag][t]) for t in 1:tlen))
    end
    ll
end
function llimporthazard(param,paths)
    @unpack cases=paths
    ll=0.0
    tlen=length(cases.imported)
    imphaz=importhazard(param)
    ll+=sum((logpdf(Poisson(imphaz[t]),cases.imported[t]) for t in 1:tlen))
    ll
end

function importhazard(param)
    @unpack h₀,r=param
    @. h₀*exp(r*((1:tlen)-Dates.value(newyeardate-initdate)-1))
end
function detectprob(param)
    @unpack q,delay=param
    @. q*(1.0-exp(-1/delay*(tlen-(1:tlen))))
end


# +
# test simulation run

nb=NBmu(2,0.5)
gt=Gmusd(7,1)
R0=2
k=0.5
h0=10.0
r=0.05
qt=fill(0.1,tlen)
nsample=100

paths=(hazard=(imported=importhazard((h₀=h0,r=r)),loc=zeros(tlen)),
        cases=(imported=zeros(Int,tlen),loc=zeros(Int,tlen)),
        infness=(imported=observed.imported.+0.0,loc=observed.loc.+0.0))

@time lls=infcasesgibbs!(paths,nsample,nb,gt,observed,qt,tlen)
@time llnbdist(params(nb),paths)
@time lldetectprob(qt,paths,observed)
@time llimporthazard((h₀=h0,r=r),paths)
using RCall;@rimport base as R
@show R.table(lls)
mean(lls)
# -

# ## MCMC sampling

# +
# unknown variables: λ, i, j, R₀, k, q
parms=Dict{Symbol,Any}(
    :h₀=>0.1,
    :r=>0.1,
    :R₀=>1.0,
    :k=>0.5,
    :nlogq=>0.5,
    :delay=>10.0
)
priors=Dict{Symbol,Any}()
for parname in keys(parms)
    priors[parname]=Stochastic(()->Uniform(0,5))
end
priors[:delay]=Stochastic(()->Uniform(0,20))
dotter=[100]
inputs=Dict{Symbol,Any}(
    :SI=>serialint,
    :observed=>observed,
    :paths=>paths,
    :zerotrick=>0.0,
    :invtemp=>1.0,
    :smcsize=>100,
    :counter=>([0],dotter)
)

inits=merge(parms,inputs)
inits=[inits]

model=Model(
    j=Logical(1,(paths,k)-> paths.cases.imported),
    i=Logical(1,(paths,k)-> paths.cases.loc),
    ll_smc=Logical(()->0.0,false),
    
    ll_nb=Logical(
        (R₀,k,paths)->llnbdist(params(NBmu(R₀,k)),paths)
        , false
    ),
    ll_q=Logical(
        (nlogq,delay,paths,observed)->begin
            qt=detectprob((q=exp(-nlogq),delay=delay))
            return(lldetectprob(qt,paths,observed))
        end
        , false),
    ll_h=Logical(
        (h₀,r,paths)->llimporthazard((h₀=h₀,r=r),paths)
        , false),
    lltotal=Logical((ll_nb,ll_q,ll_h,ll_smc,invtemp)->sum((ll_nb,ll_q,ll_h,ll_smc))*invtemp
        , false),
    zerotrick=Stochastic((lltotal)->begin Poisson(-lltotal) end,false),
    count=Logical((counter,k)->begin

            counter[1][1]
        end
    );
    priors...
)

infcasessample=Sampler(
    [:ll_smc],
    (R₀,k,h₀,r,nlogq,delay,observed,paths,counter)->begin
        paths.hazard.imported.=importhazard((h₀=h₀,r=r))
        nb=NBmu(R₀,k)
        ll=infcasesgibbs!(paths,smcsize,nb,serialint.dist,observed,qt,tlen)
        i,j=paths.cases
        #counter
        counter[1].+=1
        if counter[1][1]==counter[2][1]
            counter[1].=0
            print(".")
        end
        ll
    end
)



setsamplers!(model,[NUTS(collect(keys(parms))),infcasessample]);
setsamplers!(model,[AMM(collect(keys(parms)),Matrix{Float64}(I,6,6).*0.05),infcasessample]);



# +
mcmclen=100000
burn=div(mcmclen,5)
dotter.=div(mcmclen,100)
const smcsize=100

chain = mcmc(model, inputs, inits, mcmclen, burnin=burn, thin=div(mcmclen,1000), chains=1)
# -

showparam=[:R₀,:k,:nlogq,:delay,:h₀,:r]
@show Mamba.draw(Mamba.plot(chain[:,showparam,:]))
# Visualise
i=chain[:,:i,:].value
im=median(i,dims=1)[1,:,1]
j=chain[:,:j,:].value
jm=median(j,dims=1)[1,:,1]
q=median(hcat([detectprob((q=exp(-chain[t,:nlogq,1].value[1]),delay=chain[t,:delay,1].value[1])) for t in chain.range]...),dims=2)
inferredvars=PyPlot.plot.([jm,im,q.*100])
PyPlot.legend(getindex.(inferredvars,1),["jₜ: imported cases","iₜ: local cases","detection probability (%)"]);


@show fittodata=PyPlot.plot.([observed.imported,jm.*q[:,1],observed.loc,im.*q[:,1]])
PyPlot.legend(getindex.(fittodata,1),["Jₜ","E(Jₜ)","Iₜ","E(Iₜ)"]);
using RCall;@rimport graphics as rg;
rg.pairs(chain[:,[:nlogq,:R₀,:k,:h₀,:r],:].value[:,:,1])


