BASELINE:
vs. all norm methods:
none (base):        0.376
normSize:           0.374           -> is worse, ignore
normBbox:           0.404           -> big improvment    -> does not depend on ICC+ hyper params -> TODO: one new grid search for normBbox (only filter threshold + other best combinations) -> or other TODO use noNorm with poseline fallback
normGlac:           0.392           -> medium improvment -> p@1 depends on action centers quality and therefore on all action center hyper params -> therefore include it in grid search even if lower as normBbox

vs. all sort methods:
cr_desc (base):     0.376
nmd_desc:           0.222           -> got worse
hr_nmd_desc:        0.390           -> medium improvment

vs. fallback:   
none (base):        0.376
bisection:          0.376           -> will only change GAC not poselines -> therefore it will only change retrieval result if normGlac is used
gac norm fallback:  0.376           -> parameter for normGlac and therefore obsioisly only changes if normGlac is used
poselines:          0.418           -> big improvment

vs conebase:
0 = none (base)     0.376
0.5                 0.376           -> will only change GAC not poselines -> therefore it will only change retrieval result if normGlac is used
1                   0.376           -> will only change GAC not poselines -> therefore it will only change retrieval result if normGlac is used
1.5                 0.376           -> will only change GAC not poselines -> therefore it will only change retrieval result if normGlac is used





=> some of the ICC+ improvments are only for GAC, therefore the retrieval results will only improve for glac. We therfore use the BASELINE+normGlac as new baseline method:
normGlac dependend configurations:
vs. fallback:
none (gac base)     0.392
bisection           0.374
gac norm fallback:  0.392           -> no improvments, maybe edge case is missing. -> we will include it in final method selection since it makes sense from a logical point of view and does not make the result worse

vs conebase:
0 = none (gac base) 0.392
0.5                 0.376
1                   0.386
1.5                 0.400           -> unstable value -> we have to include it in grid search





Additional: => poseline fallback brought a good improvment because of many newly added poselines, 
therefore it makes sense to reavaluate the other methods with this fallback since they all dependend on the quality of the poselines.
new baseline: BASELINE+poselines fallback
vs. all norm methods:
                    p@1     p@1 without fallback
none (base):        0.418   (0.376)
normSize:           0.414   (0.374)     -> big improvment compared to previos runs because of the many newly added poselines possibly the problem described in Figure 3.5 c,d) decreases -> but still worse then without normalization
normBbox:           0.386   (0.404)     -> got worse compared to nombBbox without poselines fallback and also worse than fallback+noneNorm
normGlac:           0.406   (0.392)     -> medium improvment over normGlac without fallback but sligtly worse than fallback+noneNorm -> because normGlac dependend on hyperparams include it in final gridsearch

vs. all sort methods:
cr_desc (base):     0.418   (0.376)
nmd_desc:           0.206   (0.222)     -> is still the worst method -> result got even worse -> 0.2 would be random choice -> therefore nmd completly useless on it's own
hr_nmd_desc:        0.448               -> is still the best method -> even bigger improvment




Resume for final Method: 
    -> better put everthing in table and see +/- difference to baseline method for a more objective resume

sorting method:     hr_nmd_desc         -> outperformed all other sortin methods with and without poseline fallback

fallback:           poseline fallback   -> big improvement -> defenetly use it
                    gac norm fallback   -> no improvments, maybe edge case is missing. 
                                            -> we will include it in final method selection since it makes sense from a logical point of view and does not make the result worse
                    -bisection fb       -> do not use bisection fallback as it decreased the result

norm method:        none                -> was worse without poseline fallback but good with poseline fallback -> since we decide on use poseline fallback we will eventually do gridsearch for it
                    normBbox            -> was good without poseline fallback but really worse with poseline fallback -> since we decide on use poseline fallback we will not do gridsearch for it
                    normGlac            -> was good without fallback and still good with fallback -> dependend on many hyperparams, therefore defenetly grid search for this method needed
                                                -> hard deciscion what method to use else, maybe +/- difference will help
                                                    -> we will defenetly continue with normGlac, but should we include other method as well?

conebase:           1.5                 -> increasing conebase increase the result as well, should include it in final grid search as it depends on combination of all the other hyper params


finals für step 2 (gridsearch):
grid: cbs, cs, ca, co, th
hr_nmd_desc+poseline fallback+gac norm fallback+normGlac

evtl auch noch, da kleines grid
grid: th                                    -> (andere hyper param können ignoriert werden da diese nur für global action center relevant):
hr_nmd_desc+poseline fallback+noneNorm
hr_nmd_desc+normBbox


TODO: normSize und normBBox hatten wohl falschen filter threshold... -> brauchen ja viel kleineren... -> diese nochmal wiederholen