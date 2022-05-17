#!/usr/bin/env python
# coding: utf-8
# python3

import re,numpy as np, random, operator, matplotlib.pyplot as plt
import math

#Function to read the content of a .tsp file from the tsplib library

def read_tsp_data(tsp_name):
	tsp_name = tsp_name
	with open(tsp_name) as f:
		content = f.read().splitlines()
		cleaned = [x.lstrip() for x in content if x != ""]
		return cleaned


#Function to obtain the number of cities from the instance

def detect_dimension(in_list):
	non_numeric = re.compile(r'[^\d]+')
	for element in in_list:
		if element.startswith("DIMENSION"):
			return non_numeric.sub("",element)

#Function to get the list of cities
 
def get_cities(list,dimension):
	cities_set = []
	dimension = int(dimension)
	for item in list:
		for num in range(1, dimension + 1):
			if item.startswith(str(num)):
				index, space, rest = item.partition(' ')
				if rest not in cities_set:
					cities_set.append(rest)
	return cities_set


#Function to brake each coordinate to a tuple

def city_tup(list):
	cities_tups = []
	for item in list:
		first_coord, space, second_coord = item.partition(' ')
		cities_tups.append((float(first_coord.strip()), float(second_coord.strip())))
	return cities_tups

#Function to get the cities as a dictionary

def create_cities_dict(cities_tups):
	return dict(zip((range(1,len(cities_tups)+1)),cities_tups))

def read_tsp(dimension,data):
	cities_dict = create_cities_dict(city_tup(get_cities(data,dimension)))	
	return cities_dict

#Function to evaluate an individu

def evaluation(individu,cities):
	distance = 0.0
	for i in range(0, len(individu)):
		fromCity = individu[i]
		toCity = None
		if i+1 < len(individu):
			toCity = individu[i+1]
		else:
			toCity = individu[0]
		xDiff = cities.get(fromCity)[0]-cities.get(toCity)[0]
		yDiff = cities.get(fromCity)[1]-cities.get(toCity)[1]
		distance += round(np.sqrt((xDiff ** 2) + (yDiff ** 2)))
	return distance

#Function to a display a tour

def plottour(instance,individu,cities):
	plt.figure(figsize=(8, 10), dpi=100)  
	plt.title('Traveling Salesman : ' + instance)
	for point in cities.values():
		plt.plot(point[0],point[1],'ro')
	x=[]
	y=[]
	for i in range(0, len(individu)):
		x.append(cities.get(individu[i])[0])
		y.append(cities.get(individu[i])[1])

	#retour ville de départ
	x.append(cities.get(individu[0])[0])
	y.append(cities.get(individu[0])[1])

	plt.plot(x,y,color='black', lw=1)
	
	#plt.ion()
	plt.show()
	#plt.pause(5) 

def croisementRang(parent1, parent2, p):
    """Croisement à point fondé sur le rang"""
    fils = parent1[:p]
    
    for i in range(len(parent2)):
        if parent2[i] not in fils:
            fils.append(parent2[i])
            
    return fils

def mutation(element):
    """Effectue la mutation d'un élément"""
    
    while True:
        indice1 = random.randrange(0, len(element))
        indice2 = random.randrange(0, len(element))
        
        if indice1 != indice2: # On veut deux indices différents
            break
        
    temp = element[indice1]
    element[indice1] = element[indice2]
    element[indice2] = temp
    return element
    
def genetique(cities, N, NbG, pm):
    """Params : cities (dictionnaire des villes), N (taille de la population), NbG (nombre de générations), 
    pm (probabilité de mutation)
    
    Retourne l'individu de fitness maximum"""
    
    # Générer une population initiale P de N individus
    P = [random.sample(range(1, N+1), N) for i in range(N)]
    
    for k in range(1, NbG+1):
        f = []
        for i in range(1, N+1):
            f.append(evaluation(P[i-1],cities))
        Pprime = []
        
        for i in range(1, math.floor(N/2)+1):
            # Selection : Choisir deux parents dans P en utilisant les évaluations fi
            parent1 = f.index(max(f))
            f2 = list(f)
            f2[parent1] = -1 # Pour ne pas prendre le même maximum 
            parent2 = f2.index(max(f2))
            
            # Croisement : Croiser les deux parents pour obtenir deux nouveaux individus
            # Croisement à point fondé sur le rang
            fils1 = croisementRang(P[parent1], P[parent2], random.randrange(2, N-2))
            fils2 = croisementRang(P[parent2], P[parent1], random.randrange(2, N-2))
            
            # Mutation : Avec une probabilité pm, modifier les nouveaux individus
            if random.random() < pm:
                fils1 = mutation(fils1)
                fils2 = mutation(fils2)
            
            # Insertion : Insérer les nouveaux individus dans Pprime
            Pprime.append(fils1)
            Pprime.append(fils2)

        P = Pprime
    
    # Retourner l’individu de P de fitness maximum
    fFit = []
    for i in range(len(f)):
        fFit.append(1/f[i])
        
    indiceFitMax = fFit.index(max(fFit))
    return P[indiceFitMax]

def main():
    instance = "kroA100.tsp"
    data = read_tsp_data(instance)
    nbCities = int(detect_dimension(data))	
    cities = read_tsp(nbCities,data)
    
    individuOptimal=[1,47,93,28,67,58,61,51,87,25,81,69,64,40,54,2,44,50,73,68,85,82,95,13,76,33,
	37,5,52,78,96,39,30,48,100,41,71,14,3,43,46,29,34,83,55,7,9,57,20,12,27,86,35,62,60,77,23,98,91,
	45,32,11,15,17,59,74,21,72,10,84,36,99,38,24,18,79,53,88,16,94,22,70,66,26,65,4,97,56,80,31,89,42,
	8,92,75,19,90,49,6,63]
    
    # Application de l'algorithme génétique
    NbG = 100
    pm = 0.1
    individu = genetique(cities, nbCities, NbG, pm)
    #print(individu)
    
    # Représentation de l’évolution de la qualité de la meilleure solution en fonction du nombre de générati
    X = [30*i for i in range(1, 15)]
    gene = [genetique(cities, nbCities, Nb, pm) for Nb in X]
    Y = [evaluation(gene[i], cities) for i in range(len(gene))]
    plt.plot(X, Y)
    plt.title("Evolution de la qualité de la meilleure solution")
    plt.xlabel("Nombre de générations")
    plt.ylabel("Evaluation de la solution")
    plt.show()
    
    plottour(instance,gene[math.ceil(len(gene)/2)],cities)
    plottour(instance,gene[-1],cities)
    plottour(instance,individuOptimal,cities)
    
""" print(cities)
	print('Number of cities = ', nbCities)
	#Random solution
	individu = random.sample(range(1,nbCities+1), nbCities)	
	print(individu)
	print('EvaluationRd = ',evaluation(individu,cities))
	print('FitnessRd = ',1/evaluation(individu,cities))
	#Optimal solution for KroA100.tsp
	
	print('EvaluationOptimal = ',evaluation(individuOptimal,cities))
	print('FitnessOptimal = ',1/evaluation(individuOptimal,cities))	"""
    
    

main()

"""
Question 1:
    Si on fait un codage direct et un croisement à un point, il est possible qu on ne passe pas par toutes 
    les villes et qu on passe plusieurs fois par le même par ailleurs -> non souhaitable.
    
    De même pour un codage ordinal avec un croisement à un point fondé sur le rang, qui fera 
    même raccourcir le trajet du voyageur.
    
Question 2:
    Mutation : on pourrait échanger deux indices de villes dans la liste de trajet.
    Selection : on pourrait choisir deux parents avec la meilleure évaluation.
"""

