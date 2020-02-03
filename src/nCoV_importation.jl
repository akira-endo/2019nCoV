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
# \lambda_t=\lambda_0\exp(r t)
# $$
#
# We model the offspring distribution of nCoV by a negative binomial distribution $\mathrm{NB}\left(k,\frac{k}{R_0+k}\right)$, where $R$ is the mean (i.e. the basic reproduction number) and $k$ is the overdispersion parameter. The reproductive property of the negative binomial distriubution assures that the number of secondary infections caused by the overall infector at time $t$ follows $\mathrm{NB}\left(k(i_t+j_t),\frac{k}{R_0+k}\right)$, and we assume these offsprings are distributed on the timeline $t$ according to the serial interval distribution $s_\tau$.
#
# The renewal process is thus represented as
# $$
# i_t \sim \sum_{\tau=1}^\infty \mathrm{Binom}\left(\mathrm{NB}\left(k(i_{t-\tau}+j_{t-\tau}),\frac{k}{R_0+k}\right),s_\tau \right).
# $$
# Note here that the sum of distributions denotes the distribution of the summed probabilistic variables.
#
# The observation of cases is assumed to follow the binomial sampling:
# $$
# I_t\sim \mathrm{Binom}(i_t,q_t), \\
# J_t\sim \mathrm{Binom}(j_t,q_t)
# $$

# ## Setups

# +
# Packages
using Mamba, Distributions
# Constants
const tlen=180

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
# -

# ## Simulation functions

# +
using Distributions, StatsFuns, StatsBase
function importandbranch!(cases::NamedTuple{(:imported,:loc,:infness,:hazard),NTuple{4,Vector{R}}} where R<:Real,importhazard,nbparm,gtimevec,atleastone=false)
    for t in 1:length(cases.loc)
        if sum(vec[t] for vec in cases)+importhazard[t] == 0 continue end
        
        # draw cases
        cases.imported[t]+=rand(Poisson(importhazard[t]))
        cases.loc[t]+=rand(Poisson(cases.hazard[t]))
        currinfs=cases.imported[t]+cases.loc[t]
        
        # draw gamma: total offsprings
        if currinfs!=0
            α,θ=nbparm
            cases.infness[t]=rand(Gamma(α*currinfs,θ))
            #distribute infness on timeline
            cases.hazard[t+1:end].+=cases.infness[t].*gtimevec[1:length(cases.loc)-t]
            else cases.infness[t]=0.0
        end
        
        # if conditioned that infections ≧ 1
        if atleastone
            plusoneat=t+ceil(Int,sample(1:length(gtimevec),Weights(gtimevec)))
            if plusoneat<=length(cases.loc) cases.loc[plusoneat]+=1 end
            atleastone=false
        end
    end
end
# clusters generator
function nbcluster(nsample,branchdist,gtimedist,tlen,seed=1)
    logpsolo=logpdf(branchdist,0)*seed
    
    # generate chains conditioned that secondary infections ≧ 1
    clusters=[zeros(Float64,tlen) for i in 1:nsample]
    setindex!.(clusters,1,1)
    branch!.(clusters,branchdist,gtimedist,true)
    return((logpsolo=logpsolo,samples=clusters))
end

function importcluster(nsample,branchdist,gtimedist,tlen,importhazard)
    labels=(:imported,:loc,:infness,:hazard)
    samples=[NamedTuple{labels}(collect((zeros(Float64,tlen) for j in 1:4))) for i in 1:nsample]
    gtimevec=diff(cdf.(gtimedist,0:tlen))
    importandbranch!.(samples,Ref(importhazard),Ref(params(branchdist)),Ref(gtimevec))
    return(samples)
end

# +
# test simulation run
nb=NBmu(2,3)
gt=Gmusd(7,1)
R0=2
k=0.5
λ0=0.02
r=0.01

importhazard=[λ0*exp(r*t) for t in 1:tlen]
@time clustersamples=importcluster(100,NBmu(2,1),gt,tlen,importhazard);
# -

# ## MCMC sampling

# +
# unknown variables: λ, i, j, R₀, k, q
parms=Dict{Symbol,Any}(
    :λ₀=>[0.1],
    :r=>[0.1],
    :R₀=>[1.0],
    :k=>[0.5],
    :q=>ones(float64,tlen)
)
priors=Dict{Symbol,Any}()
for parname in keys(parms)
    priors[parname]=Stochastic(1,()->Uniform(0,5))
end
priors[:q]=Stochastic(1,()->Uniform(0,1))

inputs=Dict{Symbol,Any}(
    :SI=serialint
    :casedata=casedata
    :zerotrick=0.0
)

inits=copy(parms)
inits=[inits]

model=Model(
    j=Stochastic(1,
        (λ₀,r,q)->Poisson.(λ₀*exp.(r.*(1:tlen))) #conditioned to J: Gibbs
    ),
    i=Stochastic(1,
        (j,R₀,k)->0.0# NegativeBinomial.(j.*k,k/(R₀+k)): renewal process
    ),
    
    llcase=Logical(
        ()->0.0
        , false
    ),
    llctrl=Logical(
        (hhCtrl_i,hhCtrl_n,Λc,βh)->begin
            Rmat=fill(βh[5],5,5)
            Rmat[1:2,1:2].=βh[1]
            Rmat[3:4,3:4].=βh[2]
            Rmat[4,1:2]=Rmat[1:2,4].=βh[3]
            Rmat[5,5]=βh[4]
            global counter+=1
            global chk=(Rmat,Λc,βh)
            if(counter%100==0) println(counter) end
            return(hetLK.ll(hhCtrl_i,hhCtrl_n,Λc.value,Rmat,0.5))
        end
        , false
    ),
    lltotal=Logical((llcase,llctrl,invtemp)->(llcase+llctrl)*invtemp),
    pseudodata=Stochastic((lltotal)->mb.LogPosterior(lltotal.value),false),
    exnodes...
)
global counter=0
setsamplers!(model,[AMM(collect(keys(parms)),Matrix{Float64}(I,10,10).*0.001)])
sim1 = mcmc(model, inputs, inits, 3000, burnin=2000, thin=2, chains=1)
# -


