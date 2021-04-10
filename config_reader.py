import copy
import multiprocessing as mp
import os
import re
import time
import random
import json
import pynvml

pynvml.nvmlInit()

def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    ctx = mp.get_context('spawn')

    subprocess=[]
    all_gpu_queue=[0,1,2, 3]
    gpu_queue = []
    waittime = 120
    for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
        while len(gpu_queue)==0 and not run_args.cpu:
            for index in  all_gpu_queue:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used/1024/1024<100:
                    gpu_queue.append(index)
            if len(gpu_queue)==0:
                print("Waiting for Free GPU ......")
                time.sleep(waittime)
            else:
                print("Avaliable devices: ",gpu_queue)
        if len(gpu_queue)>0:
            device_id = str(gpu_queue[0])
            gpu_queue.remove(gpu_queue[0])
            run_args.device_id = device_id
        print("Using Random Seed", run_args.seed)
        if run_args.seed == -1:
            run_args.seed = random.randint(0,1000)
        p = ctx.Process(target = target, args=(run_args,))

        # debug
        # target(run_args)
        subprocess.append(p)
        p.start()
        time.sleep(1)
        if len(gpu_queue) == 0 and not run_args.cpu:
            time.sleep(waittime)
            # subprocess=[]
    list(map(lambda x:x.join(),subprocess))


def _read_config(path):
    lines = open(path).readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if v == "None":
            continue
        if v.startswith("["):
            v = v[1:-1].replace(",", "")
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))
    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    _print = (lambda x: print(x)) if verbose else lambda x: x

    if args.config:
        config = _read_config(args.config)

        for run_repeat, run_config in config:
            print("-" * 50)
            print("Config:")
            print(run_config)

            args_copy = copy.deepcopy(args)
            run_config = copy.deepcopy(run_config)
            config_list = _convert_config(run_config)
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)
            

            run_args_list = []
            # batch eval data/example/scierc_train/2021-01-25_15:53:22.993652/final_model
            if run_args.label == "batch_eval_flag":
                save_path = run_args.model_path
                # save_model_type = run_args.save_model_type
                for dirpath,dirnames,filenames in sorted(os.walk(save_path),key = lambda x:x[0]):
                    if dirpath.endswith("final_model"):
                        dataset_name=re.match(".*/(.*)_train/.*",dirpath).group(1)
                        # print(dirpath)
                        # exp_label=dirpath.split("/")[-3]
                        # exp_time=dirpath.split("/")[-2]
                        # if dataset_name=="ade" or dataset_name=="ace04":
                        #     dataset_name2=re.match(save_path+"(.*?)_train",dirpath).group(1)
                        #     run_args.label= dataset_name2+"_eval"
                        #     run_args.dataset_path ="data/datasets/"+dataset_name+"/"+dataset_name2+"_test_dep_context.json"
                        # else:
                        run_args.label = dataset_name+"_eval"
                        run_args.dataset_path = "data/datasets/"+dataset_name.split("_")[0]+"/"+dataset_name+"_test_dep_context.json"

                        run_args.model_path = dirpath
                        run_args.tokenizer_path = dirpath
                        # run_args.types_path = "data/datasets/"+dataset_name+"/"+dataset_name+"_types.json"
                        

                        args_path = "/".join(dirpath.split("/")[:-1])+"/args.json"
                        args_dict = json.load(open(args_path))
                        # print(args)
                        run_args.weight_decay = args_dict["weight_decay"]
                        run_args.types_path = args_dict["types_path"]
                        
                        run_args.model_type = args_dict["model_type"]
                        
                        # run_args.log_path = args_dict["log_path"]
                        run_args.neg_entity_count = args_dict["neg_entity_count"]
                        run_args.neg_relation_count = args_dict["neg_relation_count"]
                        if run_args.rel_filter_threshold == -1:
                            run_args.rel_filter_threshold = args_dict["rel_filter_threshold"]
                        run_args.syn_graph = args_dict["syn_graph"]
                        run_args.sema_graph = args_dict["sema_graph"]
                        run_args.fusion_rgcn = args_dict["fusion_rgcn"]
                        run_args.tw_rel_atten_token = args_dict["tw_rel_atten_token"]
                        run_args.tw_ent_atten_token = args_dict["tw_ent_atten_token"]
                        run_args.tw_rel_atten_subword = args_dict["tw_rel_atten_subword"]
                        run_args.tw_ent_atten_subword = args_dict["tw_ent_atten_subword"]
                        run_args.trigger_attn = args_dict["trigger_attn"]
                        run_args.max_span_size = args_dict["max_span_size"]
                        # run_args.eval_batch_size = args_dict["eval_batch_size"]
                        run_args.size_embedding = args_dict["size_embedding"]
                        run_args.prop_drop = args_dict["prop_drop"]
                        run_args.full_graph_retain_rate = args_dict["full_graph_retain_rate"]
                        run_args.dt_graph_retain_rate = args_dict["dt_graph_retain_rate"]
                        run_args.tw_grad_flow_token = args_dict["tw_grad_flow_token"]
                        run_args.tw_grad_flow_subword = args_dict["tw_grad_flow_subword"]
                        run_args.trigger_grad_flow = args_dict["trigger_grad_flow"]
                        run_args.seed = args_dict["seed"]
                        run_args.size_embedding = args_dict["size_embedding"]
                        run_args.store_predictions = args_dict["store_predictions"]
                        run_args.store_examples = args_dict["store_examples"]
                        run_args.sampling_processes = args_dict["sampling_processes"]
                        run_args.sampling_limit = args_dict["sampling_limit"]
                        # run_args.max_pairs = args_dict["max_pairs"]
                        run_args.split_epoch = args_dict["split_epoch"]
                        
                        run_args_list.append(copy.deepcopy(run_args))
            else:
                run_args_list.append(run_args)

            for run_args in run_args_list:
                print(run_args)
                print("Repeat %s times" % run_repeat)
                print("-" * 50)
                for iteration in range(run_repeat):
                    _print("Iteration %s" % iteration)
                    _print("-" * 50)

                    yield copy.deepcopy(run_args), run_config, run_repeat
            
            time.sleep(3)

    else:
        yield args, None, None