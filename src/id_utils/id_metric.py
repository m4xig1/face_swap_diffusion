import os
import torch
import numpy as np
import torchvision.transforms as transforms
from src.id_utils.id_modules.model_irse import IR_101
from src.id_utils.id_modules.mtcnn.mtcnn import MTCNN
import multiprocessing as mp
from tqdm import tqdm


class IDMetric:
    def __init__(
        self,
        device,
        n_threads=8,
    ):  
        self.curricular_face_path = "src/id_utils/CurricularFace_Backbone.pth"
        self.device = device
        self.n_threads = n_threads

        self.facenet = IR_101(input_size=112)
        self.facenet.load_state_dict(torch.load(self.curricular_face_path, map_location=self.device))
        self.facenet.to(self.device)
        self.facenet.eval()


    def extract_on_data(self, inp):
        inp_data, fake_data = inp
        # inp_data = [inp_data]
        # fake_data = [fake_data]
        print(inp_data)
        print(fake_data)
        id_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        count = 0

        scores = []
        for i in tqdm(range(len(inp_data))):
            count += 1

            input_im = inp_data[i]
            input_id = self.facenet(id_transform(input_im.resize((112, 112))).unsqueeze(0).to(self.device))[0]

            result_im = fake_data[i]
            result_id = self.facenet(id_transform(result_im.resize((112, 112))).unsqueeze(0).to(self.device))[0]

            score = float(input_id.dot(result_id))
            scores.append(score)
        return scores

    def __call__(
        self,
        real_data_path,
        fake_data_path,
        from_data=None,
    ):

        results = self.extract_on_data([from_data["inp_data"], from_data["fake_data"]])

        mean = np.mean(results)
        std = np.std(results)

        result_str = "New ID Average score is {:.4f}+-{:.4f}".format(mean, std)
        print(result_str)

        return mean


class IDMetric2:
    def __init__(
        self,
        n_threads=8,
    ):  
        self.curricular_face_path = "/home/aamatosyan/pers-diffusion/OurPhotoMaker/src/id_utils/CurricularFace_Backbone.pth"
        self.n_threads = n_threads
        try:
          torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
          pass 

    def get_name(self):
        return "ID"

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


    def extract_on_data(self, inp):
        inp_data, fake_data, paths = inp
        inp_data = [inp_data]
        fake_data = [fake_data]
        paths = [paths]

        facenet = IR_101(input_size=112)
        facenet.load_state_dict(torch.load(self.curricular_face_path))
        facenet.cuda()
        facenet.eval()
        mtcnn = MTCNN()
        id_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        pid = mp.current_process().name
        tot_count = len(paths)
        count = 0

        scores_dict = {}
        for i in range(len(paths)):
            count += 1

            input_im = inp_data[i]
            input_im, _ = mtcnn.align(input_im)
            if input_im is None:
                print("{} skipping {}".format(pid, paths[i]))
                continue

            input_id = facenet(id_transform(input_im).unsqueeze(0).cuda())[0]

            result_im = fake_data[i]
            result_im, _ = mtcnn.align(result_im)
            if result_im is None:
                print("{} skipping {}".format(pid, paths[i]))
                continue

            result_id = facenet(id_transform(result_im).unsqueeze(0).cuda())[0]
            score = float(input_id.dot(result_id))
            scores_dict[os.path.basename(paths[i])] = score
        return scores_dict

    def __call__(
        self,
        real_data_path,
        fake_data_path,
        from_data=None,
    ):

        pool = mp.Pool(self.n_threads)
        zipped = zip(
            from_data["inp_data"], from_data["fake_data"], from_data["paths"]
        )
        results = pool.map(self.extract_on_data, zipped)


        scores_dict = {}
        for d in results:
            scores_dict.update(d)

        all_scores = list(scores_dict.values())
        mean = np.mean(all_scores)
        std = np.std(all_scores)

        result_str = "New ID Average score is {:.3f}+-{:.3f}".format(mean, std)
        print(result_str)

        return scores_dict, mean, std