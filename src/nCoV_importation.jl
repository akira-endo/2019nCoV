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
# Let $S_i$ and $C_i$ be the dates of symptom onset and reporting of nCoV case $i$ outside China.
# Let $q_i$ be the symptomatic ratio of nCoV infection (potentially specific to individual $i$) and $g(\tau)$ be the distribution of the generation time (i.e. time between the linked infection pairs) of nCoV.
# $$
#
# $$

# ## Precompute transmission chains

using Distributions
using StatsFuns
function branch!(timeseries::Vector{<:Real},branchdist,gtimedist,atleastone=false)
    for t in 1:length(timeseries)
        if timeseries[t]==0 continue end
        
        # count secondary infections
        if typeof(branchdist)<:NegativeBinomial
            sumbranchdist=NegativeBinomial(params(nb)[1]*timeseries[t],params(nb)[2])
            newinfs=rand(sumbranchdist)
        else
            newinfs=sum(rand(branchdist) for n in 1:timeseries[t])
        end
        
        # if conditioned that infections ≧ 1
        if atleastone
            newinfs+=1
            atleastone=false
        end
        
        # switch to deterministic model if > 10000 new infections
        if newinfs>10000
            timeseries[t+1:end].+=newinfs*diff(cdf.(gtimedist,0:length(timeseries)-t))
        else
        
        # distribute on timeseries
        for inf in 1:newinfs
            tnextgen=t+ceil(Int,rand(gtimedist))
            if tnextgen<=length(timeseries)
                timeseries[tnextgen]+=1 end
            end
        end
        timeseries[t]=round(timeseries[t])
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

# +
NBmu(mu,k)=NegativeBinomial(k,mu/(mu+k))
Gmusd(mu,sd)=Gamma(mu^2/sd^2, sd^2/mu)

nb=NBmu(2,1)
gt=Gmusd(7,1)

murange=0.05:0.1:1
krange=0.05:0.1:2
@time clustersamples=[nbcluster(1000,NBmu(mu,k),gt,180,1) for mu in murange for k in krange];
# -

# ## Conditional sampling


