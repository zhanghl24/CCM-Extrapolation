import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.constants import PI
import numpy as np
import sionna
sionna.config.xla_compat=True

class Scatter_info():
    
    def __init__(self, roughness_table, em_prop_table, tile_length, tile_width, transform):
        '''
        Describe a single scatter.
        roughness_table: Record roughness parameter in different regions.
        em_prop_table: Record complex permittivity in different regions.
        tile_length: 1*1 float. Length of single tile.
        transform: 2*3 tensor. The first row is the position [x, y, z] of the center
                of the scatter. The second row is the rotation Eular angle [α, β, γ]
                of the scatter. 
        '''
        self.roughness_table = roughness_table
        self.em_prop_table = em_prop_table
        self.tile_length = tile_length
        self.tile_width = tile_width
        self.transform = transform
        self.prop = "scat"

    @classmethod
    def gen_table(cls, mat_type_table, roughness, em_prop):
        '''
        mat_type_table: n*n tensor. For a scatter surface contains materials 
                    of m types, elements in mat_type_table take value in [0, 1, ..., m-1]
        roughness: m*1 tensor. Each element stands for roughness parameter lambda 
                of one kind of material.
        em_prop: m*1 tensor. Each element stands for complex permittivity ita
                of one kind of material.
        '''
        return tf.gather(roughness, mat_type_table), tf.gather(em_prop, mat_type_table)
    
class Em_property():
    def __init__(self, scatter_infos, id_map):
        '''
        Defines all em parameters in the propagation scene.
        scatter_infos: dicts contains k [object_id, Scatter_info] pairs. k is the number of scatters.
        id_hash: map object_id to the indice of the same scatterer in scatter_infos.
        '''
        self.id_map = id_map

        self.table_shape = np.zeros((len(scatter_infos), 2),dtype=np.int32)
        self.tile_size = np.zeros((len(scatter_infos), 2),dtype=np.float32)
        max_row = max([item.roughness_table.shape[0] for item in scatter_infos])
        max_col = max([item.roughness_table.shape[1] for item in scatter_infos])
        self.roughness_table = np.ones([len(scatter_infos), max_row, 
                                        max_col], dtype=np.float32) * -1
        self.em_prop_table = np.ones([len(scatter_infos), max_row, 
                                        max_col], dtype=np.complex64) * -1
        self.shift = np.zeros([len(scatter_infos), 3], dtype=np.float32)
        self.rotmat = np.zeros([len(scatter_infos), 3, 3], dtype=np.float32)


        for ii in range(len(scatter_infos)):
            self.roughness_table[ii,0:scatter_infos[ii].roughness_table.shape[0], 
                                 0:scatter_infos[ii].roughness_table.shape[1]]\
            = scatter_infos[ii].roughness_table.numpy()
            self.em_prop_table[ii,0:scatter_infos[ii].em_prop_table.shape[0], 
                                 0:scatter_infos[ii].em_prop_table.shape[1]]\
            = scatter_infos[ii].em_prop_table.numpy()
            self.shift[ii] = scatter_infos[ii].transform[0].numpy()
            rotation = scatter_infos[ii].transform[1].numpy()
            R_z = np.array([[np.cos(rotation[0]), -np.sin(rotation[0]), 0],
                            [np.sin(rotation[0]), np.cos(rotation[0]), 0],
                            [0, 0, 1]], dtype=np.float32)
            R_y = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])],
                            [0, 1, 0],
                            [-np.sin(rotation[1]), 0, np.cos(rotation[1])]],
                            dtype=np.float32)
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(rotation[2]), -np.sin(rotation[2])],
                            [0, np.sin(rotation[2]), np.cos(rotation[2])]],
                            dtype=np.float32)
            self.rotmat[ii] = R_z @ R_y @R_x
            self.table_shape[ii] = scatter_infos[ii].roughness_table.shape
            self.tile_size[ii] = [scatter_infos[ii].tile_length, scatter_infos[ii].tile_width]

        self.roughness_table = tf.convert_to_tensor(self.roughness_table)
        self.em_prop_table = tf.convert_to_tensor(self.em_prop_table)
        self.shift = tf.convert_to_tensor(self.shift)
        self.rotmat = tf.convert_to_tensor(self.rotmat)
        self.table_shape = tf.convert_to_tensor(self.table_shape)
        self.tile_size = tf.convert_to_tensor(self.tile_size)
    
    #@tf.function(jit_compile=True)
    def __call__(self, points, object_ids, prop):
        '''
        points: [batch_dims, 3] tensor. Positions of the intersection points.
        object_ids: [batch_dims] tensor. Integers uniquely identifying the intersected objects.
        prop: decide which param to return: "scat" for roughness, "em" for complex permittivity
        '''
        shape = object_ids.shape
        points = tf.reshape(points, [-1, 3])
        object_ids = tf.reshape(object_ids, [-1])  
        object_ids = tf.gather(self.id_hash, object_ids)
        shift = tf.gather(self.shift, object_ids)
        rotmat = tf.gather(self.rotmat, object_ids)
        points = tf.matmul(rotmat, tf.expand_dims(points - shift, axis=-1))
        points = tf.reshape(points, [-1, 3])[:, 0:2]
        ind_shift = tf.gather(self.table_shape, object_ids)
        ind_shift = tf.cast(ind_shift, dtype=tf.float32)
        tile_size = tf.gather(self.tile_size, object_ids)
        ind = tf.cast(tf.math.floor(points / tile_size \
                                        + ind_shift / 2), tf.int32)
        ind = tf.concat([tf.expand_dims(object_ids, axis=-1), ind], axis=1)

        if prop == "scat":
            result = tf.gather_nd(self.roughness_table, ind)
        elif prop == "em":
            result = tf.gather_nd(self.em_prop_table, ind)
        return tf.reshape(result, shape)
    
    def update_roughness(self, roughness_table_list): # for learning parameters
        shape = self.roughness_table.shape[1:]
        for ii in range(len(roughness_table_list)):
            roughness_table_list[ii] = tf.pad(roughness_table_list[ii],
                                              paddings=[[0, shape[0] - roughness_table_list[ii].shape[0]],
                                                        [0, shape[1] - roughness_table_list[ii].shape[1]]],
                                              constant_values=-1)
        self.roughness_table = tf.stack(roughness_table_list)
        
class Scatter(Layer):
    '''
    For a scatter surface composed of multiple materials, this callable 
    class specifies the roughness distribution on its surface.
    '''
    def __init__(self, em_property): 
        super(Scatter, self).__init__()
        self.em_property = em_property
        
    def build(self, input_shape):
        return 0
    
    #@tf.function(jit_compile=True)
    def call(self, object_id, points, k_i, k_s, n_hat): 
        alpha_r = self.em_property(points, object_id, "scat")
        t_r = alpha_r ** 0.5 * (1.6988 * alpha_r ** 2 + 10.8438 * alpha_r)\
              / (alpha_r ** 2 + 6.2201 * alpha_r + 10.2415)
        a_u_r = 2 * PI / alpha_r * (1 - tf.exp(-alpha_r))
        a_b_r = 2 * PI / alpha_r * tf.exp(-2 * alpha_r) * (tf.exp(alpha_r) - 1)
        dot_k_i_n = tf.reduce_sum(tf.multiply(k_i, n_hat), axis=-1)
        k_r = k_i - 2 * tf.multiply(n_hat, tf.expand_dims(dot_k_i_n, axis=-1))
        cosbeta_r = tf.reduce_sum(tf.multiply(k_r, n_hat), axis=-1)
        s_r = (tf.exp(t_r) * tf.exp(t_r * cosbeta_r) - 1) \
                / ((tf.exp(t_r) - 1) * (tf.exp(t_r * cosbeta_r) + 1))
        a_r = a_u_r * s_r + a_b_r * (1 - s_r)
        cosbeta_r = tf.reduce_sum(tf.multiply(k_r, k_s), axis=-1)
        f_s = 1 / a_r * tf.exp(alpha_r * (cosbeta_r-1))
        f_s = tf.reshape(f_s, object_id.shape)
        return f_s
    
class Radio_material(Layer):
    '''
    For a scatter surface composed of multiple materials, this callable 
    class specifies the complex permittivity distribution on its surface.
    '''
    def __init__(self, em_property):
        super(Radio_material, self).__init__()
        self.em_property = em_property
    
    def build(self, input_shape):
        return 0
    
    #@tf.function(jit_compile=True)
    def call(self, object_id, points):
        temp = self.em_property(points, object_id, "em")
        return temp, tf.ones_like(temp, dtype=tf.float32), tf.zeros_like(temp, dtype=tf.float32)

def cal_size(scatter_length, scatter_width, corr_len):
    tile_row = np.int32(np.ceil(scatter_length / corr_len)*3)
    tile_colomn = np.int32(np.ceil(scatter_width / corr_len)*3)
    tile_length = scatter_length / tile_row
    tile_width = scatter_width / tile_colomn
    return tile_row, tile_colomn, tile_length, tile_width

def cal_R(tile_row, tile_colomn, tile_length, tile_width, corr_len):
    '''
    calculate correlation matrix used in scatterer modeling.
    '''
    tile_num = tile_row * tile_colomn
    tile_length = tf.constant(tile_length, dtype=tf.float32)
    tile_width = tf.constant(tile_width, dtype=tf.float32)
    corr_len = tf.constant(corr_len, dtype=tf.float32)

    temp = tf.tile(tf.range(tile_num), [tile_num])
    temp = tf.reshape(temp, [tile_num, tile_num])
    x2 = tf.cast(temp // tile_row, tf.float32)
    y2 = tf.cast(temp % tile_colomn, tf.float32)
    temp = tf.transpose(temp)
    x1 = tf.cast(temp // tile_row, tf.float32)
    y1 = tf.cast(temp % tile_colomn, tf.float32)
    dist = tf.math.sqrt(((x1-x2)*tile_length)**2 + ((y1-y2)*tile_width)**2)
    corr_mat = tf.exp(-(dist/corr_len)**2)
    eig_value, eig_vector = tf.linalg.eigh(corr_mat)
    eig_value = eig_value[::-1]
    eig_vector = tf.reverse(eig_vector, axis=[1])
    sum = 0
    total_sum = tf.reduce_sum(eig_value)
    for ii in range(len(eig_value)):
        sum = sum + eig_value[ii]
        if sum >= 0.8 * total_sum:
            break
    R = eig_vector[:,0:ii+1] @ tf.linalg.diag(eig_value[0:ii+1])**0.5
    v_trainable = tf.random.normal(shape=(1, ii+1), mean=0.0, stddev=1.0)
    v_trainable = tf.Variable(v_trainable, dtype=tf.float32)
    return R, v_trainable