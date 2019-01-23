import json

env_info_dict = {

'Pendulum' : {
'gym_env_name' : 'Pendulum-v0',
'state_labels' : ['cos(ang)', 'sin(ang)', 'ang_vel'],
'action_labels' : ['torque'],
'action_space_type' : 'continuous'
},

'LunarLander' : {
'gym_env_name' : 'LunarLander-v2',
'state_labels' : ['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'v_ang'],
'action_labels' : ['nothing', 'engine_L', 'engine_main', 'engine_R'],
'action_space_type' : 'discrete'
},

'CartPole' : {
'gym_env_name' : 'CartPole-v0',
'state_labels' : ['pos_cart', 'v_cart','pole_angle', 'v_poletip'],
'action_labels' : ['cart_L', 'cart_R',],
'action_space_type' : 'discrete'
},


}


fname = 'gym_env_info.json'

with open(fname, 'w') as outfile:
    json.dump(env_info_dict, outfile, indent=4)









#
