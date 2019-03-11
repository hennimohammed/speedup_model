import numpy as np 

class Loop_Iterator():
        def __init__(self, it_id, dict_repr, depth=0):
            self.depth=depth
            #get loop iterator
            iterator = next(it for it in dict_repr['iterators']['iterators_array'] if it['it_id'] == it_id)

            self.id = it_id
            self.lower_bound = iterator['lower_bound']
            self.upper_bound = iterator['upper_bound']

        def __repr__(self):
            return f"({self.lower_bound}, {self.upper_bound})"

        def __array__(self):
            return [self.id, self.lower_bound, self.upper_bound]

class Input():
    def __init__(self, input_id, dict_repr, depth=0):
        self.id = input_id
        self.depth = depth
        
        #search for input_id
        input_ = next(i for i in dict_repr['inputs']['inputs_array']
                                if i['input_id'] == input_id)

        self.dtype = input_['data_type']
    
    def __repr__(self):
        return f"Input {self.id}"

    def __array__(self):
        return [self.id]

class Access_pattern():
    def __init__(self, access_matrix, depth=0):
        self.access_matrix = np.array(access_matrix)
        self.max_shape = (4, 5)
    
    def __repr__(self):
        return repr(self.access_matrix)
    
    def __array__(self):
        rows = self.max_shape[0] - self.access_matrix.shape[0]
        cols = self.max_shape[1] - self.access_matrix.shape[1]

        for _ in range(cols):
            self.access_matrix = np.insert(self.access_matrix, -1, 0, axis=1)

        for _ in range(rows):
            self.access_matrix = np.insert(self.access_matrix, len(self.access_matrix), 0, axis=0)

        
        return self.access_matrix.flatten()

        
        





class Computation():
    def __init__(self, comp_id, dict_repr, depth=0):
        self.depth = depth
        self.id = comp_id
        self.max_children = 17 #max accesses
        self.max_comp_len = 21*self.max_children
        #search for comp_i
        computation = next(c for c in dict_repr['computations']['computations_array']
                                if c['comp_id'] == comp_id)

        self.dtype = computation['lhs_data_type']

        self.op_histogram = computation['operations_histogram'][0] #take only first row for now

        self.children = []

        mem_accesses = computation['rhs_accesses']['accesses']

        for mem_access in mem_accesses:
            inp = Input(mem_access['comp_id'], dict_repr, self.depth+1)
            access_pattern = Access_pattern(mem_access['access'], self.depth+1)

            self.children.append((inp, access_pattern))


    def __repr__(self):
        sep = '\n' +  (self.depth+1)*'\t'

        children_repr = [repr(child) for child in self.children]
        children_repr = sep + sep.join(children_repr)

        return f"Computation {self.id}:" + children_repr

    def __array__(self):
        children_arr = []

        for child in self.children[:self.max_children]:
            inp = child[0]
            access = child[1]

            children_arr.extend(inp.__array__() + access.__array__())

        children_arr.extend([-1] * (self.max_comp_len - len(children_arr)))
        
        return children_arr
        
        


        
class Loop():
    def __init__(self, loop_repr, dict_repr, depth=0):
        self.depth = depth
        self.id = loop_repr['loop_id']

        it_id = loop_repr['loop_it']
        self.iterator = Loop_Iterator(it_id, dict_repr)

        #search and create all children of loop (other loops, and computation assignments)
        self.children_dict = {}

        #add assignments to children
        comps = loop_repr['assignments']['assignments_array']
        for comp in comps:
            comp_id = comp['id']
            position = comp['position']

            self.children_dict[position] = Computation(comp_id, dict_repr, self.depth+1)

        #add loops to children
        loops = dict_repr['loops']['loops_array']

        for loop in loops: 
            if loop['parent'] == self.id:
                self.children_dict[loop['position']] = Loop(loop, dict_repr, self.depth+1)

        self.children = self.sort_children()
        
    def sort_children(self):
        #sort children by position 
        return list(list(zip(*sorted(self.children_dict.items(), key=lambda x: int(x[0]))))[1])  

    def __repr__(self):
        children_repr = [repr(child) for child in self.children]
        children_repr = '\n' + (self.depth+1)*'\t'  + "\n".join(children_repr)
        #print(children_repr)

        return  f"Loop {self.id} {repr(self.iterator    )}:" + children_repr

    def __array__(self):
        loop_arr = []
        loop_arr.extend(self.iterator.__array__())

        if not isinstance(self.children[0], Loop): 
            #fill loop space with -1
            loop_arr_len = len(loop_arr)
            loop_arr.extend([-1]*loop_arr_len * (3 - self.depth))
        
       
        loop_arr.extend(self.children[0].__array__())

        return loop_arr

        
        

    

class Loop_AST():

    def __init__(self, name, dict_repr=None):

        self.name = name
        self.root_loop = None
        self.dtype_int_dict = {"p_int": 199}
        self.load_from_dict(dict_repr)


    
    def dtype_to_int(self, dtype):
        return self.dtype_int_dict[dtype]


    def load_from_dict(self, dict_repr):
        if not dict_repr:
            return
            
        self.dict_repr = dict_repr

        loops = dict_repr['loops']['loops_array']

        #find root loop
        root = next(l for l in loops if l['parent'] == -1)

        self.root_loop = Loop(root, dict_repr)

    def __array__(self):
        return np.array(self.root_loop.__array__())