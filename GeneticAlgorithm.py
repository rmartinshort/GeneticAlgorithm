#!/usr/bin/env python 
#RMS 2018

#A genetic algorithm to aid in feature selection in machine learning problems

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

class GeneticAlgorithm:

    def __init__(self,X,Y,Algorithm,Niter=100,keep_fraction=0.5,mutation_rate="auto",nfeatures="auto",test_size=0.3,njobs=1):

        '''

        A simple genetic algorithm designed to work with sklearn objects.

        The object of the algorithm is to select the optimal combination of columns of input datafame X that produce the
        best prediction of Y given the algorithm object provided.

        To do this, the algorithm starts by randomly selecting different 
        combinations of columns. These become the first 'generation' of individuals

        For each combination, the sklearn algorithm is trained in the associated columns and tested. The test score 
        becomes the 'fitness' of that combination/individual

        Individuals are then selected for 'breeding'. Those with the highest fitness scores have the highest chances
        of doing so. When two individuals 'breed'. Two children are produced by a simple crossover of their column IDs

        The child generation is then combined with the 'best' of the parent generation via the 'keep_fraction' argument

        The algorithm proceeds by testing the fitness of each generation. The result is an 'optimal' combination of 
        features that correspond to the best fitness score. 

        This should work with any supervised sklearn object


        Inputs:
        X - dataframe containing the predictors only
        Y - datafame or series containing the target only
        Algorithm - sklearn classifier or regression object, such as an RandomForestClassifier
        Niter - number of iterations of the genetic algorithm
        keep_fraction - the proportion of fittest parents to keep in each new generation
        mutation_rate - the probability of mutation in each child
        nfeatures - the maximum number of features that an output model can have
        test_size - test_size in the train_test_split that occurs in model fitness evaluation 

        Main outputs:
        self.fitness_evolution - list of the best fitness value from each generation
        self.best_individual_evolution - list of arrays of the best individuals in each generation
        self.feature_selection - dataframe corresponding to the selected features 
        self.best_fitness - fitness score correspondng to self.feature_selection
        self.best_individual - individual corresponding to self.feature_selection

        '''


        self.dataset = X
        self.response = Y
        self.algorithm = Algorithm #needs to be a sklearn object
        self.Niter = Niter #number of iterations 
        self.parent_keep = keep_fraction
        self.test_size = test_size
        self.nprocs = int(njobs)

        if self.nprocs > mp.cpu_count():

            raise ValueError("Entered number of processes > CPU count!")


        self.feature_columns = self.dataset.columns

        if nfeatures == 'auto':
            self.nfeatures = len(self.feature_columns)
        else:
            self.nfeatures = nfeatures

        self.P = 2*int(np.ceil(self.nfeatures*1.5/2)) #number of individuals in a given generation

        if mutation_rate == 'auto':

            self.mutation_rate = 1.0/(self.P*np.sqrt(self.nfeatures))
        else:
            self.mutation_rate = mutation_rate

        self.fitness_evolution = []
        self.best_individual_evolution = []

        #These three things are typically the most desired output
        self.feature_selection = None
        self.best_fitness = None
        self.best_individual = None


    def fitness(self,generation):

        '''
        Assess the fitness of a generation of individuals

        This is the part that takes a long time because it must train a supervised ML algorithm on all individuals in a generation
        '''

        def determine_fitness(subgeneration,output,pos):

            fitness_array = np.zeros(np.shape(subgeneration)[0])

            for i in range(np.shape(subgeneration)[0]):
            
                individual = subgeneration[i,:]
                
                #Subset the columns based on this individual
                X_individual = self.dataset[[self.dataset.columns[j] for j in range(len(individual)) if individual[j] == 1]]
                
                #Split into train-test datasets
                X_train, X_test, y_train, y_test = train_test_split(X_individual,self.response,test_size=self.test_size)
                
                #Fit the classifier
                self.algorithm.fit(X_train,y_train)
                
                #Report fitness score (score in the testing dataset)
                fitness = self.algorithm.score(X_test,y_test)
                
                #append to fitness array
                fitness_array[i] = fitness

            output.put((pos,fitness_array))


        process_output = mp.Queue()
        subarrays = np.array_split(generation,self.nprocs)
        processes = [mp.Process(target=determine_fitness,args=(subarrays[i],process_output,i)) for i in range(self.nprocs)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        results = [process_output.get() for p in processes]
        results.sort()
        results = np.array([r[1] for r in results]).flatten()
       
        return results

    def make_new_generation(self,old_generation,old_fitness_array):
        
        '''
        Make a new generation of individuals
        '''
        
        generation_size = len(old_fitness_array)
            
        #Vector describing the probability of reporduction of each individual in a generation
        prob_weights = 2*np.argsort(old_fitness_array/(generation_size*(generation_size+1)))[::-1]
        
        prob_reproduction = prob_weights/np.sum(prob_weights)
        
        #Make vector of indices to choose
        a = np.arange(generation_size)
        
        children = np.zeros([2*generation_size,np.shape(old_generation)[1]])
        
        for i in range(generation_size):
            parent_index_pair = np.random.choice(a,size=2,replace=False,p=prob_reproduction)
            
            parent1 = old_generation[parent_index_pair[0]]
            parent2 = old_generation[parent_index_pair[1]]
            
            #Do cross over and apply mutation to generate two children for each parent pair
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            #Generate locations of genetic information to swap
            pos = np.random.choice(len(parent1),size=int(len(parent1)/2),replace=False)
            child1[pos] = parent2[pos]
            child2[pos] = parent1[pos]
            
            #Generate mutation vector
            mutate1 = np.random.binomial(1,self.mutation_rate,len(parent1))
            mutate2 = np.random.binomial(1,self.mutation_rate,len(parent1))
            
            #Generate children and fill child array
            child1 = (child1+mutate1 >= 1).astype(int)
            child2 = (child2+mutate2 >= 1).astype(int)
            
            children[i,:] = child1
            children[-(i+1),:] = child2
            
        #shuffle and return only the same number of children as there were parents 
        np.random.shuffle(children)
        
        new_generation = children[0:generation_size,:]
        
        #replace some fraction of the children with the fittest parents, if desired
        
        nparents_to_keep = int(self.parent_keep*generation_size)
        
        if nparents_to_keep > 0:
            parents_keep = np.argsort(old_fitness_array)[::-1][:nparents_to_keep]

            for i in range(len(parents_keep)):
                new_generation[i,:] = old_generation[parents_keep[i],:]

        np.random.shuffle(new_generation)
        
        
        return new_generation 

    def fit(self):

        '''
        Run the genetic algorithm to obtain the optimal features for this problems
        This part takes a long time and could be parallelized
        '''

        #Make the first generation 
        old_generation = np.zeros([self.P,self.nfeatures])
        for i in range(self.P):
            old_generation[i,:] = np.random.binomial(1,0.5,self.nfeatures)

        old_fitness_array = self.fitness(old_generation)

        self.best_fitness = np.max(old_fitness_array)
        self.best_individual = old_generation[np.argmax(old_fitness_array),:]

        self.fitness_evolution.append(self.best_fitness)
        self.best_individual_evolution.append(self.best_individual)

        for n in range(1,self.Niter):

            print("GeneticAlgorithm: Testing generation %i" %n)

            #Make new generation
            new_generation = self.make_new_generation(old_generation,old_fitness_array)
            #Get fitness of new generation
            new_fitness_array = self.fitness(new_generation)

            #Locate and extract the best individual and its score
            self.best_fitness = np.max(new_fitness_array)
            self.best_individual = new_generation[np.argmax(new_fitness_array),:]
            self.fitness_evolution.append(self.best_fitness)
            self.best_individual_evolution.append(self.best_individual)

            old_fitness_array = new_fitness_array
            old_generation = new_generation

        #Get the features associated with the 'winning' individual

        self.feature_selection = self.dataset[[self.dataset.columns[j] for j in range(len(self.best_individual)) if self.best_individual[j] == 1]]



def split_array(array,nprocs):

    '''Split an array into chunks for each process to work on'''




        


