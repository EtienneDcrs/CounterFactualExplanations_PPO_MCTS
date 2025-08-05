# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:37:55 2020

@author: Iacopo
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import manhattan_distances as L1
from sklearn.metrics.pairwise import euclidean_distances as L2
from skimage.metrics import structural_similarity as SSIM
from tqdm import tqdm


class CERTIFAI:
    def save_counterfactuals_to_csv(self, filename):
        """
        Save all generated counterfactuals to a CSV file and a CSV with the original samples.
        If there is no counterfactual for a sample, it will include the original sample and 0 in the column 'counterfactual_found', instead of 1
        Each row contains only the counterfactual features of the sample and if it's a counterfactual.
        """
        if self.results is None or len(self.results) == 0:
            print("No counterfactuals to save.")
            return

        counterfactual_rows = []
        original_rows = []
        for sample_idx, (sample, counterfacts, distances) in enumerate(self.results):
            orig = sample.values[0] if hasattr(sample, 'values') else sample[0]
            # Save original sample
            original_row = {col: orig[idx] for idx, col in enumerate(sample.columns)}
            original_row['sample_id'] = sample_idx
            original_rows.append(original_row)

            if len(counterfacts) == 0:
                # No counterfactuals: include original features with counterfactual_found=0
                row = {col: orig[idx] for idx, col in enumerate(sample.columns)}
                row['sample_id'] = sample_idx
                row['counterfactual_found'] = 0
                counterfactual_rows.append(row)
            else:
                # Save each counterfactual with counterfactual_found=1
                for cf in counterfacts:
                    row = {col: cf[idx] for idx, col in enumerate(sample.columns)}
                    row['sample_id'] = sample_idx
                    row['counterfactual_found'] = 1
                    counterfactual_rows.append(row)

        # Create DataFrames
        counterfactual_df = pd.DataFrame(counterfactual_rows)
        original_df = pd.DataFrame(original_rows)

        # Save to CSVs
        counterfactual_df.to_csv(filename, index=False)
        original_filename = f"{os.path.splitext(filename)[0]}_original.csv"
        original_df.to_csv(original_filename, index=False)
        print(f"Counterfactuals saved to {filename}")
        print(f"Original samples saved to {original_filename}")

    def __init__(self, Pm = .2, Pc = .5, dataset_path = None,
                 numpy_dataset = None, label_encoders = None, scaler = None):
        """The class instance is initialised with the probabilities needed
        for the counterfactual generation process and an optional path leading
        to a .csv file containing the training set. If the path is provided,
        the class will assume in some of its method that the training set is tabular
        in nature and pandas built-in functions will be used in several places, instead
        of the numpy or self defined alternatives."""
        
        self.Pm = Pm
        self.Pc = Pc
        self.Population = None
        self.distance = None
        self.label_encoders = label_encoders
        self.scaler = scaler
        
        if dataset_path is not None:
            self.tab_dataset = pd.read_csv(dataset_path)
        else:
            self.tab_dataset = None
            if numpy_dataset is not None:
                self.tab_dataset = numpy_dataset
            
        self.constraints = None
        self.predictions = None
        self.results = None
    
    @classmethod
    def from_csv(cls, path, label_encoders=None, scaler=None):
        return cls(dataset_path=path, label_encoders=label_encoders, scaler=scaler)
    
    def change_Pm(self, new_Pm):
        '''Set the new probability for the second stage of counterfactual
        generation.
        Arguments:
            Inputs:
                new_Pm: new probability of picking a sample from the counterfactuals
                to be changed as described in the original paper for the second
                step of the generation process.
                
            Outputs:
                None, the Pm attribute is changed'''
        
        self.Pm = new_Pm
        
    def change_Pc(self, new_Pc):
        '''Set the new probability for the third stage of counterfactual
        generation.
        Arguments:
            Inputs:
                new_Pc: new probability of picking a sample from the counterfactuals
                to be changed as described in the original paper for the third
                step of the generation process.
                
            Outputs:
                None, the Pc attribute is changed'''
        
        self.Pc = new_Pc
        
    def get_con_cat_columns(self, x):
        
        assert isinstance(x, pd.DataFrame), 'This method can be used only if input\
            is an instance of pandas dataframe at the moment.'
        
        con = []
        cat = []
        
        for column in x:
            if x[column].dtype == 'O':
                cat.append(column)
            else:
                con.append(column)
                
        return con, cat
        
    def Tab_distance(self, x, y, continuous_distance = 'L1', con = None,
                     cat = None):
        """Distance function for tabular data
        """
        
        assert isinstance(x, pd.DataFrame), 'This distance can be used only if input\
            is a row of a pandas dataframe at the moment.'
            
        
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns = x.columns.tolist())
        else:
            y.columns = x.columns.tolist()
        
        if con is None or cat is None:
            con, cat = self.get_con_cat_columns(x)
        
        if len(cat)>0:
            
            cat_distance = len(cat) - (x[cat].values == y[cat].values).sum(axis = 1)
        
        else:
            
            cat_distance = 1
            
        if len(con)>0:
            
            if continuous_distance == 'L1':
                con_distance = L1(x[con], y[con])
                
            else:
                con_distance = L2(x[con], y[con])
                
        else:
            con_distance = 1
            
        return len(con)/x.shape[-1]*con_distance + len(cat)/x.shape[-1]*cat_distance
    
    def img_distance(self, x, y):
        
        distances = []
        
        for counterfact in y:
            distances.append(SSIM(x, counterfact))
        
        return np.array(distances).reshape(1,-1)
            
    def calculate_distance(self, original_features, modified_features, con=None, cat=None):
        """
        Calculate L1 (Manhattan) distance between encoded original and modified features.
        """
        original_df = pd.DataFrame([original_features], columns=self.tab_dataset.columns)
        modified_df = pd.DataFrame([modified_features], columns=self.tab_dataset.columns)
        
        if con is None or cat is None:
            con, cat = self.get_con_cat_columns(original_df)
        
        dist = 0
        for column in original_df.columns:
            if column in cat:
                # Categorical: count as 1 if values differ
                dist += float(original_df[column].values[0] != modified_df[column].values[0])
            else:
                # Continuous: compute absolute difference, with fallback for non-numeric
                try:
                    dist += abs(float(original_df[column].values[0]) - float(modified_df[column].values[0]))
                except (ValueError, TypeError):
                    # Fallback for non-numeric continuous features
                    dist += float(original_df[column].values[0] != modified_df[column].values[0])
        
        return dist
        
    def calculate_sparsity(self, original_features, modified_features, con=None, cat=None):
        """
        Calculate sparsity (number of modified features) between original and modified features.
        """
        original_df = pd.DataFrame([original_features], columns=self.tab_dataset.columns)
        modified_df = pd.DataFrame([modified_features], columns=self.tab_dataset.columns)
        
        if con is None or cat is None:
            con, cat = self.get_con_cat_columns(original_df)
        
        sparsity = 0
        for column in original_df.columns:
            if column in cat:
                # Categorical: count as changed if values differ
                sparsity += float(original_df[column].values[0] != modified_df[column].values[0])
            else:
                # Continuous: count as changed if absolute difference is significant
                try:
                    diff = abs(float(original_df[column].values[0]) - float(modified_df[column].values[0]))
                    sparsity += float(diff > 1e-6)
                except (ValueError, TypeError):
                    # Fallback for non-numeric continuous features
                    sparsity += float(original_df[column].values[0] != modified_df[column].values[0])
        
        return sparsity
        
    def set_distance(self, kind = 'automatic', x = None):
        """Set the distance function to be used in counterfactual generation.
        The distance function can either be manually chosen by passing the 
        relative value to the kind argument or it can be inferred by passing the
        'automatic' value.
        
             Outputs:
                None, set the distance attribute as described above."""
        
        
        if kind == 'automatic':
            
            assert x is not None or self.tab_dataset is not None, 'For using automatic distance assignment,\
                the input data needs to be provided or the class needs to be initialised with a csv file!'
            
            if x is None:
                x = self.tab_dataset
            
            if len(x.shape)>2:
                self.distance = self.img_distance
                print('SSIM distance has been set as default')
                
            else:
                con, cat = self.get_con_cat_columns(x)
                if len(cat)>0:
                    self.distance = self.Tab_distance
                    print('Tabular distance has been set as default')
                    
                else:
                    self.distance = L1
                    print('L1 norm distance has been set as default')
                
        elif kind == 'tab_distance':
            self.distance = self.Tab_distance
        elif kind == 'L1':
            self.distance = L1
        elif kind == 'SSIM':
            self.distance = self.img_distance
        elif kind == 'L2':
            self.distance = L2
        elif kind == 'euclidean':
            self.distance = L2
        else:
            raise ValueError('Distance function specified not recognised:\
                             use one of automatic, L1, SSIM, L2 or euclidean.')
        
    def set_population(self, x=None):
        """Set the population limit (i.e. number of counterfactuals created at each generation).
        following the original paper, we define the maximum population as the minum between the squared number of features
        to be generated and 30000.
        
        Arguments:
            Inputs:
                x (numpy.ndarray or pandas.DataFrame): the training set or a sample from it, so that the number of features can be obtained.
            
            Outputs:
                None, the Population attribute is set as described above
        """
        
        if x is None:
            assert self.tab_dataset is not None, 'If input is not provided, the class needs to be instatiated\
                with a csv file, otherwise there is no input data for inferring population size.'
            
            x = self.tab_dataset
        
        if len(x.shape)>2:
            self.Population = min(sum(x.shape[1:])**2, 30000)
        else:
            self.Population = min(x.shape[-1]**2, 30000)
        
    def set_constraints(self, x = None, fixed = None):
        '''Set the list of constraints for each input feature, whereas
        each constraint consist in the minimum and maximum value for 
        the given continuous feature. If a categorical feature is encountered,
        then the number of unique categories is appended to the list instead.
            '''
        
        fixed_feats = set() if fixed is None else set(fixed)
        
        self.constraints = []
        
        if x is None:
            x = self.tab_dataset
        
        if len(x.shape)>2:
            x = self.tab_dataset if self.tab_dataset is not None else x.copy()
            
            x = pd.DataFrame(x.reshape(x.shape[0], -1))
            
        if isinstance(x, pd.DataFrame):
            for i in x:
                if i in fixed_feats:
                    # Placeholder if the feature needs to be kept fixed in generating counterfactuals
                    self.constraints.append(i)
                # Via a dataframe is also possible to constran categorical fatures (not supported for numpy array)
                elif x.loc[:,i].dtype == 'O':
                    self.constraints.append((0, len(pd.unique(x.loc[:,i]))))
                else:
                    self.constraints.append((min(x.loc[:,i]), max(x.loc[:,i])))
        
        else:
            assert x is not None, 'A numpy array should be provided to get min-max values of each column,\
                or, alternatively, a .csv file needs to be supplied when instatiating the CERTIFAI class'
            
            for i in range(x.shape[1]):
                if i in fixed_feats:
                    # Placeholder if the feature needs to be kept fixed in generating counterfactuals
                    self.constraints.append(i)
                else:
                    self.constraints.append((min(x[:,i]), max(x[:, i])))
                
    def transform_x_2_input(self, x, pytorch=True):
        '''Function to transform the raw input in the form of a pandas dataset
        or of a numpy array to the required format as input of the neural net(s)
        
        Arguments:
            Inputs:
                x (pandas.DataFrame or numpy.ndarray): the "raw" input to be
                transformed.
                
                pytorch (bool): the deep learning library
                used for training the model analysed. Options are torch==True for 
                pytorch and torch==False for tensorflow/keras
                
            Outputs:
                transformed_x (torch.tensor or numpy.ndarray): the transformed
                input, ready to be passed into the model.'''
                
        if isinstance(x, pd.DataFrame):
            x = x.copy()
            con, cat = self.get_con_cat_columns(x)
            
            if len(cat) > 0 and self.label_encoders is not None:
                for feature in cat:
                    if feature in self.label_encoders:
                        try:
                            x[feature] = self.label_encoders[feature].transform(x[feature])
                        except ValueError as e:
                            print(f"Warning: Unknown category in column {feature}. Using fit_transform instead.")
                            enc = LabelEncoder()
                            x[feature] = enc.fit_transform(x[feature])
                    else:
                        enc = LabelEncoder()
                        x[feature] = enc.fit_transform(x[feature])
            
            if self.scaler is not None:
                feature_columns = x.columns[:-1]  # Exclude target column
                x[feature_columns] = self.scaler.transform(x[feature_columns])
            
            # Exclude the target column (last column) to match model's expected input
            model_input = torch.tensor(x.iloc[:, :-1].values, dtype=torch.float) if pytorch else x.iloc[:, :-1].values
            
        elif isinstance(x, np.ndarray):
            if self.scaler is not None:
                x = x.copy()
                x[:, :-1] = self.scaler.transform(x[:, :-1])  # Exclude target column
            # Exclude the target column (last column) to match model's expected input
            model_input = torch.tensor(x[:, :-1], dtype=torch.float) if pytorch else x[:, :-1]
        else:
            raise ValueError("The input x must be a pandas dataframe or a numpy array")
            
        return model_input
    
    def generate_prediction(self, model, model_input, pytorch=True, classification=True):
        '''Function to output prediction from a deep learning model
        
        Arguments:
            Inputs:
                model (torch.nn.Module or tf.Keras.Model): the trained deep learning
                model.
                
                model_input (torch.tensor or numpy.ndarray): the input to the model.
                
                pytorch (bool): whether pytorch or keras is used.
                
                classification (bool): whether a classification or regression task is performed.
                
            Output:
                prediction (numpy.ndarray): the array containing the single greedily predicted
                class (in the case of classification) or the single or multiple predicted value
                (when classification = False).
        '''
        
        if classification:
            if pytorch:
                with torch.no_grad():
                    logits = model(model_input, apply_softmax=False)
                    prediction = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                prediction = np.argmax(model.predict(model_input), axis=-1)
        else:
            if pytorch:
                with torch.no_grad():
                    prediction = model(model_input).cpu().numpy()
            else:
                prediction = model.predict(model_input)
                
        return prediction
    
    def generate_counterfacts_list_dictionary(self, counterfacts_list,
                                              distances, fitness_dict,
                                              retain_k, start=0):
        '''Function to generate and trim at the same time the list containing
        the counterfactuals and a dictionary having fitness score
        for each counterfactual index in the list. 
        '''
                
        gen_dict = {i:distance for i, 
                                distance in enumerate(distances)}
                    
        gen_dict = {k:v for k,v in sorted(gen_dict.items(),
                                          key = lambda item: item[1])}
        
        selected_counterfacts = []
        
        k = 0
        
        for key,value in gen_dict.items():
            
            if k==retain_k:
                break
            selected_counterfacts.append(counterfacts_list[key])
            
            fitness_dict[start+k] = value
            
            k+=1
            
        return selected_counterfacts, fitness_dict
    
    def generate_cats_ids(self, dataset = None, cat = None):
        '''Generate the unique categorical values of the relative features
        in the dataset.
        '''
        if dataset is None:
            assert self.tab_dataset is not None, 'If the dataset is not provided\
            to the function, a csv needs to have been provided when instatiating the class'
            
            dataset = self.tab_dataset
            
        if cat is None:
            con, cat = self.get_con_cat_columns(dataset)
            
        cat_ids = []
        for index, key in enumerate(dataset):
            if key in set(cat):
                cat_ids.append((index,
                                len(pd.unique(dataset[key])),
                                pd.unique(dataset[key])))
        return cat_ids
    
    def generate_candidates_tab(self, sample, normalisation=None, constrained=True, has_cat=False, cat_ids=None, img=False):
        nfeats = sample.shape[-1]
        
        if normalisation is None:
            if constrained:
                generation = []
                temp = []
                for constraint in self.constraints:
                    if not isinstance(constraint, tuple):
                        # Repeat the fixed value for the whole population
                        fixed_value = sample.loc[:, constraint].values[0]
                        temp.append(np.full((self.Population, 1), fixed_value))
                    else:
                        temp.append(np.random.randint(constraint[0]*100, (constraint[1]+1)*100,
                                                    size=(self.Population, 1))/100)
                generation = np.concatenate(temp, axis=-1)
            else:
                low = min(sample)
                high = max(sample)
                generation = np.random.randint(low, high+1, size=(self.Population, nfeats))
        
        elif normalisation == 'standard':
            generation = np.random.randn(self.Population, nfeats)
        
        elif normalisation == 'max_scaler':
            generation = np.random.rand(self.Population, nfeats)
        
        else:
            raise ValueError('Normalisation option not recognised: choose one of "None", "standard" or "max_scaler".')
        
        if has_cat:
            assert cat_ids is not None, 'If categorical features are included in the dataset, the relative cat_ids (to be generated with the generate_cats_ids method) needs to be provided to the function.'
            generation = pd.DataFrame(generation, columns=sample.columns.tolist())
            for idx, ncat, cat_value in cat_ids:
                random_indeces = np.random.randint(0, ncat, size=self.Population)
                random_cats = [cat_value[feat] for feat in random_indeces]
                generation.iloc[:, idx] = random_cats
            distances = self.distance(sample, generation)[0]
        else:
            distances = self.distance(sample, generation)[0]
            generation = pd.DataFrame(generation, columns=sample.columns.tolist())  # Set column names here
        
        for i in sample:
            col_dtype = sample[i].dtype
            # If the column is integer, but values are strings, convert to float first
            if np.issubdtype(col_dtype, np.integer):
                # Try to convert to float first if needed, then round and cast to int
                try:
                    generation[i] = np.round(generation[i].astype(float)).astype(int)
                except Exception:
                    # Fallback: leave as is
                    pass
            elif np.issubdtype(col_dtype, np.floating):
                try:
                    generation[i] = generation[i].astype(float)
                except Exception:
                    pass
            else:
                try:
                    generation[i] = generation[i].astype(col_dtype)
                except Exception:
                    pass
        
        return generation.values.tolist(), distances
    
    def mutate(self, counterfacts_list):
        '''Function to perform the mutation step from the original paper
        
        Arguments:
            Input:
                counterfacts_list (list): the candidate counterfactuals
                from the selection step.
                
            Output:
                mutated_counterfacts (numpy.ndarray): the mutated candidate
                counterfactuals.'''
        
        nfeats = len(counterfacts_list[0])
        
        dtypes = [type(feat) for feat in counterfacts_list[0]]
        
        counterfacts_df = pd.DataFrame(counterfacts_list)
        
        random_indeces = np.random.binomial(1, self.Pm, len(counterfacts_list))
        
        mutation_indeces = [index for index, i in enumerate(random_indeces) if i]
        
        for index in mutation_indeces:
            mutation_features = np.random.randint(0, nfeats, 
                                                  size = np.random.randint(1, nfeats))
            
            for feat_ind in mutation_features:
                if isinstance(counterfacts_df.iloc[0, feat_ind], str):
                    counterfacts_df.iloc[index, feat_ind] = np.random.choice(
                        np.unique(counterfacts_df.iloc[:, feat_ind]))
                    
                else:
                    counterfacts_df.iloc[index, feat_ind] = 0.5*(
                        np.random.choice(counterfacts_df.iloc[:, feat_ind]) +
                    np.random.choice(counterfacts_df.iloc[:, feat_ind]))
        
        for index, key in enumerate(counterfacts_df):
            counterfacts_df[key] = counterfacts_df[key].astype(dtypes[index])
        
        return counterfacts_df.values.tolist()
    
    def crossover(self, counterfacts_list, return_df = False):
        '''Function to perform the crossover step from the original paper
        
        Arguments:
            Input:
                counterfacts_list (list): the candidate counterfactuals
                from the mutation step.
                
            Output:
                crossed_counterfacts (numpy.ndarray): the changed candidate
                counterfactuals.'''
        
        nfeats = len(counterfacts_list[0])
        
        random_indeces = np.random.binomial(1, self.Pc, len(counterfacts_list))
        
        mutation_indeces = [index for index, i in enumerate(random_indeces) if i]
        
        counterfacts_df = pd.DataFrame(counterfacts_list)
        
        while mutation_indeces:
            
            individual1 = mutation_indeces.pop(np.random.randint(0, len(mutation_indeces)))
            
            if len(mutation_indeces)>0:
                
                individual2 = mutation_indeces.pop(np.random.randint(0, len(mutation_indeces)))
                
                mutation_features = np.random.randint(0, nfeats, 
                                                      size = np.random.randint(1, nfeats))
                
                features1 = counterfacts_df.iloc[individual1, mutation_features]
                
                features2 = counterfacts_df.iloc[individual2, mutation_features]
                
                counterfacts_df.iloc[individual1, mutation_features] = features2
                
                counterfacts_df.iloc[individual2, mutation_features] = features1
        
        if return_df:
            return counterfacts_df
        
        return counterfacts_df.values.tolist()
    
    def fit(self, 
            model,
            x = None,
            model_input = None,
            pytorch = True,
            classification = True,
            generations = 3, 
            distance = 'automatic',
            constrained = True,
            class_specific = None,
            select_retain = 1000,
            gen_retain = 500,
            final_k = 1,
            normalisation = None,
            fixed = None,
            verbose = False):
        '''Generate the counterfactuals for the defined dataset under the
        trained model. The whole process is described in detail in the
        original paper.
        '''
        cfes = []
        not_found = 0
        
        if x is None:
            assert self.tab_dataset is not None, 'Either an input is passed into\
            the function or a the class needs to be instantiated with the path\
                to the csv file containing the dataset'
            
            x = self.tab_dataset
            
        else:
            
            x = x.copy() 
            
        if self.constraints is None:
                self.set_constraints(x, fixed)
            
        if self.Population is None:
                self.set_population(x)
                
        if self.distance is None:
                self.set_distance(distance, x)
                
        if model_input is None:
            model_input = self.transform_x_2_input(x, pytorch = pytorch)
            
        if pytorch:
            model.eval()
        
        if self.predictions is None: 
            self.predictions = self.generate_prediction(model, model_input,
                                                pytorch=pytorch,
                                                classification=classification)
        
        if len(x.shape)>2:
            
            x = x.reshape(x.shape[0], -1)
        
        self.results = []
        
        if isinstance(x, pd.DataFrame):
                
            con, cat = self.get_con_cat_columns(x)
            
            has_cat = True if len(cat)>0 else False
            
            cat_ids = None
            
            if has_cat:
                cat_ids = self.generate_cats_ids(x)
        
        else:
            x = pd.DataFrame(x)   
        
        if classification and class_specific is not None:
            x = x.iloc[self.predictions == class_specific]
            self.class_specific = class_specific
                
        tot_samples = tqdm(range(100)) if verbose else range(x.shape[0])
        
        for i in tot_samples:
            
            if verbose:
                tot_samples.set_description('Generating counterfactual(s) for sample %s' % i)
            
            sample = x.iloc[i:i+1,:]
            
            counterfacts = []
            
            counterfacts_fit = {}
            
            for g in range(generations):
            
                generation, distances = self.generate_candidates_tab(sample,
                                                                normalisation,
                                                                constrained,
                                                                has_cat,
                                                                cat_ids)
                    
                selected_generation, _ = self.generate_counterfacts_list_dictionary(
                    counterfacts_list = generation,
                    distances = distances, 
                    fitness_dict = {},
                    retain_k = select_retain, 
                    start=0)
                
                selected_generation = np.array(selected_generation)
                
                mutated_generation = self.mutate(selected_generation)
                
                crossed_generation = self.crossover(mutated_generation, 
                                                    return_df = True)
                
                gen_input = self.transform_x_2_input(crossed_generation,
                                                    pytorch = pytorch)
                
                counter_preds = self.generate_prediction(model,
                                                        gen_input,
                                                        pytorch,
                                                        classification)
                
                diff_prediction = [counter_pred!=self.predictions[i] for
                                counter_pred in counter_preds]
                
                final_generation = crossed_generation.loc[diff_prediction]
                
                if len(final_generation) == 0:
                    not_found += 1
                    continue
                
                final_distances = self.distance(sample, final_generation)[0]
                
                final_generation, counterfacts_fit = self.generate_counterfacts_list_dictionary(
                    counterfacts_list = final_generation.values.tolist(),
                    distances = final_distances, 
                    fitness_dict = counterfacts_fit,
                    retain_k = gen_retain, 
                    start = len(counterfacts_fit))
                
                counterfacts.extend(final_generation)
                
                assert len(counterfacts)==len(counterfacts_fit), 'Something went wrong: len(counterfacts): {}, len(counterfacts_fit): {}'.format(len(counterfacts), len(counterfacts_fit))
            
            counterfacts, fitness_dict = self.generate_counterfacts_list_dictionary(
                    counterfacts_list = counterfacts,
                    distances = list(counterfacts_fit.values()), 
                    fitness_dict = {},
                    retain_k = final_k, 
                    start = 0)
            
            self.results.append((sample, counterfacts, list(fitness_dict.values())))
        
        # Calculate mean distance and mean sparsity
        distances = []
        sparsities = []
        valid_samples = 0
        
        con, cat = self.get_con_cat_columns(x)
        
        for sample, counterfacts, _ in self.results:
            if len(counterfacts) > 0:
                valid_samples += 1
                for counterfact in counterfacts:
                    dist = self.calculate_distance(sample.values[0], counterfact, con, cat)
                    sparsity = self.calculate_sparsity(sample.values[0], counterfact, con, cat)
                    distances.append(dist)
                    sparsities.append(sparsity)
        
        if valid_samples > 0:
            mean_distance = np.mean(distances) if distances else 0.0
            mean_sparsity = np.mean(sparsities) if sparsities else 0.0
            print(f"Mean Distance: {mean_distance:.4f} (across {valid_samples} samples with valid counterfactuals)")
            print(f"Mean Sparsity: {mean_sparsity:.4f} features changed (across {valid_samples} samples with valid counterfactuals)")
        else:
            print("No valid counterfactuals found for any samples.")