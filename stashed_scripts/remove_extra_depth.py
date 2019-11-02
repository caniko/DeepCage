for v in axis_vectors:
    for loc in range(len(axis_vectors[v])):
		if loc == 0:
		    for k in axis_vectors[v][loc].keys():
			    for i in range(len(axis_vectors[v][loc][k])):
				    axis_vectors[v][loc][k][i] = axis_vectors[v][loc][k][i][0]
	    else:
		    for coord in range(len(axis_vectors[v][loc])):
                axis_vectors[v][loc][coord] = axis_vectors[v][loc][coord][0]