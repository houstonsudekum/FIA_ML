
## see https://github.com/sglyon/econtools/blob/master/Python/econtools/metrics.py ##
# must put in pandas columns

import pandas
import numpy
import linearmodels
from scipy import stats


def matchit(df,treatment, prscore,resamp = False,threshold = 1.0):
    """
    
    Note : this function will reset the index and drop the old index in the new dataframe taht is output
    
    
    
    df -- pandas dataframe that contains treated and non-treated samples and must have standard indexing (may update indexing)
    
    treatment -- treatment is your treatment column (should be 1 or 0)
    
    prscore -- is your propensity score column from logistic or probit regression
    
    resamp -- indicates whether we want to resample in mathcing or not
    
    threshold -- set to 1.0 so that every observation gets a match, but can be changed; threshold meaning that
    every value is matched because a propensity score can not be greater than 1 or less than 0 in our
    specification
            
            * note that if the threshold is changed and value does not meet threshold, as is, the treated and non-treated
            values are dropped from final df output (may change this)
    
    """
    
    newdf = df.reset_index(drop = True)
    
    
    if len(df[treatment].unique()) > 2:
        raise ValueError('there are too many treatment groups')
        
    for i in df[treatment].unique():
        try:
            int(i)
        except NameError:
            raise(ValueError('treatment variable is not integer'))
    
    dict1 = df.loc[:,treatment].to_dict()
    dict2 = df.loc[:,prscore].to_dict()
    vals = df.loc[:,treatment].unique()
    
    g1 = {}
    g2 = {}
    
    # creating dictionaries to store the indexes and propensity scores
    # for the treated and non-treated samples
    
    for i in dict1:
        treat = dict1[i]
        score = dict2[i]
        if treat == vals[0]:
            g1.update({i:score})
        elif treat == vals[1]:
            g2.update({i:score})
    
    # this is making sure that the more less-frequent treatment is
    # matched against the more-frequent
    
    if len(g2) > len(g1):
        g1, g2 = g2, g1
        
    # intitilize a new dictionary
    
    matching = {}
    
    # iterate through the less frequent treatment which we assume is the treatment
    
    for i in g2.keys():
        
        #pr score of treated
        
        score2 = g2[i]
        
        # set number to 0
        
        num = 0
        
        # iterate through the more frequent treatment
        
        for k in g1.keys():
            
            # pr score of non-treated
            
            score1 = g1[k]
            
            # absolute value of the distance
            
            dist = abs(score2 - score1)
            
            # if distance is below threshold
            
            if dist <= threshold:
                
                # add one to num every time a value is below threshold
                # note that num starts out at 0
                
                num = num + 1
                
                # if it is the second value or greater below threshold
                
                if num > 1:
                    test = min(matching[i].keys())
                    if dist < test:
                        
                        # reset index dict to the new values
                        
                        matching[i] = {dist:k}
                        
                        # update the delete key everytime until we get minimum values
                        
                        deletekey = k
                        
                # if it is the first value below threshold
                elif num == 1:
                    
                    # initialize dict
                    
                    matching[i] = {}
                    
                    # update dict with first value
                    
                    matching[i].update({dist:k})
                    
                    # save delete key
                    
                    deletekey = k

        # if resampling is false we need to delete the value saved in the dictionary
        # we saved that values key as deletekey
        
        if resamp == False:
            
            # if we found any matches
        
            if num > 0:
                
                # delete value from dict
                
                del g1[deletekey]
                
                # we reset the delete key everytime we find a match
                
                del deletekey
    
    # make another list to store the matched keys
    
    matched = []
    
    # iterate through the dictionary that was created
    
    for i in matching.keys():
        key1 = i
        for k in matching[i].keys():
            key2 = matching[i][k]
        
        # store keys 
        
        matched.append(key1)
        matched.append(key2)
    
    # incase resampling is set to true this makes sure each observation is unique in the final df
    
    val1 = len(matched)
    matched = list(set(matched))
    val2 = len(matched)
    
    
    if (resamp == False) & (val1 != val2):
        raise(ValueError('something is wrong'))
    
    finaldf = newdf.iloc[matched,:]
    
    return(finaldf)
        
    if len(newdf) == len(matched):
        print("""Dataframe length matches the list length; each plot is matched 1 : 1 
if a match exists below thereshold -- {}""".format(threshold))
    elif (resamp == False) & (len(newdf) != len(matched)):
        raise(ValueError('something is wrong'))
            
        
    return finaldf


def hausman(fe, re):
    """
    Compute hausman test for fixed effects/random effects models
    b = beta_fe
    B = beta_re
    From theory we have that b is always consistent, but B is consistent
    under the alternative hypothesis and efficient under the null.
    The test statistic is computed as
    z = (b - B)' [V_b - v_B^{-1}](b - B)
    The statistic is distributed z \sim \chi^2(k), where k is the number
    of regressors in the model.
    Parameters
    ==========
    fe : statsmodels.regression.linear_panel.PanelEffectsResults
        The results obtained by using sm.PanelLM with the
        method='within' option.
    re : statsmodels.regression.linear_panel.RandomEffectsResults
        The results obtained by using sm.PanelLM with the
        method='swar' option.
    Returns
    ==========
    chi2 : float
        The test statistic
    df : int
        The number of degrees of freedom for the distribution of the
        test statistic
    pval : float
        The p-value associated with the null hypothesis
    Notes
    =====
    The null hypothesis supports the claim that the random effects
    estimator is "better". If we reject this hypothesis it is the same
    as saying we should be using fixed effects because there are
    systematic differences in the coefficients.
    
    """

    # Pull data out
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov

    
    # not sure about this note but it shouldnt matter for us because
    # we are not estimating plot specific effects... and the only difference
    # between the models is the inclusion of an extra fixed effect
    # and fgls for random effects
    
                        # NOTE: find df. fe should toss time-invariant variables, but it
                        #       doesn't. It does return garbage so we use that to filter

                        #df = b[np.abs(b) < 1e8].size

    df = b.size
    
    # compute test statistic and associated p-value
    
    chi2 = numpy.dot((b - B).T, numpy.linalg.inv(v_b - v_B).dot(b - B))
    
    # from scipy library
    
    pval = stats.chi2.sf(chi2, df)

    return chi2, df, pval

class model_out(object):
    
    """
    this function tests for the inclusion of all effects with built in F-tests and
    then tests for the use of fixed or random effects with a standard hausman test;
    the default regressions used to test fixed or random effects include all effects
    
    output is the final model that will either be an OLS regression with fixed dummy variables; 
    a mixed effects regression with fixed dummy variable and random effects; or a random effects 
    regression with no dummy variables
    
    note that random effects is estimated for the entity_effect which is the first index of a
    pandas df
    
    the entity index is only important for the random effects models unless we specify EntityEffects
    for the fixed effects models
    
    """
    
    
    
    def __init__(self,df,response_var,formula,entity_index,numeric_effects,non_num_effects):
        
        # these are al inputs
        
        self.df = df
        self.response_var = response_var
        self.formula = formula
        self.entity_index = entity_index
        self.numeric_effects = numeric_effects
        self.non_num_effects = non_num_effects
        
        ###########################
        ### Get the model below ###
        ###########################
    

        testing_effects = self.numeric_effects + self.non_num_effects

        factor = len(testing_effects) - 1

        modkeys = {}

        for idx,i in enumerate(testing_effects):

            # add response variable

            newform = '{} ~ '.format(response_var) + self.formula

            temp = self.df

            temp.loc[:,'invyrtemp'] = temp.invyr

            temp = self.df.set_index([i,'invyrtemp'])


            # get values from testing effects that we will add as dummy variables

            keeps = [j for j in testing_effects if (j != i) & (j in list(temp))]

            # format the strings

            adding = ' + '.join(['C({})'.format(j) for j in keeps])

            # add to formula

            newform = newform + ' + ' + adding 


            feform = newform + ' + EntityEffects'


            for categ in keeps:
                temp.loc[:,categ] = pandas.Categorical(temp.loc[:,categ])


            REmod = linearmodels.RandomEffects.from_formula(newform,temp).fit()

            # use_lsdv makes sure that they are modeled as dummy variables (default is False)

            FEmod = linearmodels.PanelOLS.from_formula(feform,temp).fit(use_lsdv = True)

            # add to dictionary

            modkeys[i] = {'f':FEmod.f_pooled.pval,'HA':hausman(FEmod,REmod)[2]}

        # get signficant effects

        f_results = [i for i in modkeys.keys() if modkeys[i]['f'] < .1]

        # get maximum value from HA test out of all effects

        HA_max = max([modkeys[i]['HA'] for i in modkeys.keys()])

        # get index from maximum value if meeting other requirements and model others as fixed effects

        random_effects = [i for i in modkeys.keys() if (modkeys[i]['HA'] > .1)&(i in f_results)&(modkeys[i]['HA'] == HA_max)]

        # get remaining indexes to model as fixed effects

        fixed_effects = [i for i in f_results if i not in random_effects]

        # create a new df

        temp = self.df

        # save a temporary year index

        temp.loc[:,'invyrtemp'] = temp.invyr

        # make new formula

        newform = '{} ~ '.format(response_var) + formula

        if len(fixed_effects) > 0:

            # add the dummy lsdv for fixed effects

            adding = ' + '.join(['C({})'.format(j) for j in fixed_effects])

            # make new formula    

            newform = newform + ' + ' + adding

        #print(newform)

        if len(random_effects) > 0:

            temp = temp.set_index([random_effects[0],'invyrtemp'])

            FINALmod = linearmodels.RandomEffects.from_formula(newform,temp).fit()

            temp = df.set_index([entity_index,'invyrtemp'])

            if len(fixed_effects) > 0:
                mod = 'Mixed Effects'
            else:
                mod = 'Random Effects'

        else:

            # use entity index because does not matter for specification

            temp = temp.set_index([entity_index,'invyrtemp'])

            FINALmod = linearmodels.PanelOLS.from_formula(newform,temp).fit()

            mod = 'Fixed Effects'

        out = FINALmod

        
        ###########################
        ## Save the model output ##
        ###########################
        
        self.out = out
        self.mod = mod
        
        self.fixed = fixed_effects
        self.random = random_effects
    
    def printit(self):
        if len(self.fixed) > 0:    
            print("Fixed Effects : {}".format(self.fixed))
        else:
            print("No Fixed Effects")
        if len(self.random) > 0:
            print("Random Effects : {}".format(self.random))
        else:
            print("No Random Effects")
    
        print("""
    
    
                                 {}
    
    
        """.format(self.mod))
        print(self.out.summary)

    def fixed_effects(self):
        
        # if p is true we print
        #if len(self.fixed) > 0:    
        #    print("Fixed Effects : {}".format(self.fixed))
        #else:
        #    print("No Fixed Effects")
        return(self.fixed)
    
    def random_effects(self):
        #if len(self.random) > 0:
        #    print("Random Effects : {}".format(self.random))
        #else:
        #    print("No Random Effects")
        return(self.random)

    def model_type(self):
        print(self.mod)        
                

class compile():
    
    def __init__(self,mods):
        
        self.mods = mods
    
    def reg_to_df(self, long = True):

        tempdict = {}
        mod_out = [j.out for j in self.mods]
        out = [[j.response_var,dict(zip(list(i.params.index),zip(list(i.params),list(i.pvalues))))] for i,j in zip(mod_out,self.mods)]
        dfs =[[k[0],pandas.DataFrame.from_dict(k[1], orient='index', columns=['params','pvales']).reset_index()] for k in out]
        
        output = list()
        
        for name,df in dfs:
                        
            if long == False:

                names = [i+'_'+name for i in list(df) if i.lower() != 'index']
                
            else:
                
                df.loc[:,'model'] = name
                
                names = [i for i in list(df) if i.lower() != 'index']
                
                
            df.columns = ['xvars'] + names

            output.append(df)
        
        if long == True:
            
            
            df = pandas.concat(output)
        
        else:
            
            for idx,i in enumerate(output):
                if idx == 0:
                    df = i
                else:
                    df = df.merge(i,on='xvars',how='outer')      

        return(df)
    
    def stat_to_df(self):
        
        mod_out = [j.out for j in self.mods]
        out = {j.response_var:[i.rsquared,i.f_statistic_robust.stat,i.f_statistic_robust.pval,i.df_resid] for i,j in zip(mod_out,self.mods)}
        out['row_index'] = ['r-squared','f-statistic','p-value','df-residual']
        df = pandas.DataFrame.from_dict(out).set_index('row_index')
        return(df)
    
    def panel_specification(self):
        out = {j.response_var:[j.random_effects(),j.fixed_effects()] for j in self.mods}
        out['row_index'] = ['random','fixed']
        df = pandas.DataFrame.from_dict(out).set_index('row_index')
        return(df)