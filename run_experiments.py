from src.evaluate_llm import evaluate_ollama_llm
import os
import traceback
import itertools

def build_triples(model_ids, pstrats, ptypes):
    return list(itertools.product(model_ids, pstrats, ptypes))


if __name__ == '__main__':
    
    model_ids = ['llama3', 'phi3:3.8b', 'commmand-r', 'gemma:7b', 'aya:8b', 'llama3:70b']
    pstrats = ['direct', 'ec', 'ecr', 'ech']
    ptypes = ['zero_shot', 'one_shot', 'few_shot']

    triples = build_triples(model_ids=model_ids, pstrats=pstrats, ptypes=ptypes)
    print(f'Running configurations: {triples}')
    for model_id, pstrat, ptype in triples:
        out_path = os.path.join('out/jp2wz08', model_id, pstrat + "_" + ptype)

        try:
            os.makedirs(out_path)
        except FileExistsError:
            print(f'Directory {out_path} already exists. Caution: files may be overwritten.')
            pass
        
        try:
            evaluate_ollama_llm(
                model_id=model_id,
                pstrat=pstrat,
                ptype=ptype,
                training_data_path='data/human_annotated_jp2wz08/transformers/merged_jobpostings.trfds',
                out_path = out_path,
                verbose=True,
                plot=True,
                save=True
            )
        except Exception as e:
            exception_file_path = f"{out_path}/exception.txt"
            with open(exception_file_path, 'w') as f:
                f.write(f"Caught exception: {e}\n")
                traceback.print_exception(type(e), e, e.__traceback__, file=f)

            print(f"Failed: {model_id}-{pstrat}-{ptype}")
            traceback.print_exception(type(e), e, e.__traceback__)

            print(f"Failed: {model_id}-{pstrat}-{ptype}")
            traceback.print_exception(type(e), e, None)
