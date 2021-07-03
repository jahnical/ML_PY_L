from math import factorial

def combinations(iter, c):
    combs = []
    def _combinations(indices):
        #Add the combination to combs list
        combs.append([iter[i] for i in indices])

        #Check if last index is the last in iter
        if (indices[-1] >= len(iter)-1):
            #Increase a previous index
            increased_previous = False
            for i in range(len(indices)-2, -1, -1):
                #Check if the index can be increased and still be in range
                if (indices[i]+(len(indices)-i) < len(iter)):
                    #Increase index i and set those after i to indices immidiate after i
                    indices[i] += 1
                    counter_index = indices[i] + 1
                    for q in range(i+1, len(indices)):
                        indices[q] = counter_index
                        counter_index += 1
                    increased_previous = True
                    break
            #Return -1 if indices cannot be increased further
            if not increased_previous: return

        #Else increase the last index
        else: indices[-1] += 1

        #Recursively call _combinations
        _combinations(indices)

    _combinations([i for i in range(c)])

    #combs_count = factorial(len(iter))/(factorial(c)*factorial(len(iter)-c))
    return combs

# count, length, combs = combinations(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 6)
# print(count)
# print(length)
# print(combs)