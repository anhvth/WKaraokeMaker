
from avcv.all import *
from kmaker.dataloader import *
from tools.make_karaoke_video import *
from tools.test_submit import *


class Predictor:
    def __init__(self, ckpt_path='pretrained/whisper_karaoke_base_vn.pth', sot=True):
        self.eval_model = load_eval_model(ckpt_path, sot=sot)
        self.collate_fn = collate_fn_with_sot if sot else collate_fn_without_sot

    def predict(self, json_file):
        item, batch = preproc(json_file, self.collate_fn)
        with torch.inference_mode():
            outputs = self.eval_model.forward_both(
                        batch['inputs'],
                        labels=batch['labels'],
                        ctc_labels=batch['w2v_labels'],
                    )
            bboxes = outputs['bbox_pred'][batch['dec_pos']]

        xyxy = box_cxcywh_to_xyxy(bboxes)[:,[0,2]]
        words = [_[0] for _ in item.words]
        
        results = []
        xyxy = xyxy.clip(0, 1)
        for (x1,x2), word in zip((xyxy*30).tolist(), words):
            results.append((word, x1, x2))
        results = [Segment(*result, 1) for result in results]
        
        results = convert_result_to_competion_format(results, json_file, 1000)   
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("path_to_output")
    parser.add_argument('sample_json')
    parser.add_argument('--audio_file', default=None)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--no-sot', dest='sot', action='store_false')
    args = parser.parse_args()

    
    # for json_file in json_files:
    json_file = args.sample_json
    predictor = Predictor(sot=args.sot)

    
    fname = os.path.basename(json_file).replace(".json", "")
    if args.audio_file is not None:
        audio_file = args.audio_file
    else:
        audio_file = get_audio_file(json_file)
    
    output_video = '/tmp/karaoke'
    pred_path = os.path.join(output_video, fname + '.json')
    
    results = predictor.predict(json_file)
    mmcv.dump(results, pred_path)
    if args.output_file is not None:
        output_video_path = args.output_file
    else:
        output_video_path = os.path.join(
            output_video,
            os.path.basename(json_file).replace(".json", ".mp4"),
        )
    
    make_karaoke_video(pred_path, audio_file, output_video_path)
    print('-> {}'.format(osp.abspath(output_video_path)))
