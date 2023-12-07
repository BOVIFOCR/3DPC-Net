import sys
import torch
import torch.nn as nn
import numpy as np

try:
    from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
except:
    from iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200



class _3DPCNet(nn.Module):
    def __init__(self, encoder_name='r18', decoder_type='decoder_embedd_multitask1', face_embedd_size=256, num_output_points=2500, num_axis=3, fp16=False, **kwargs):
        super(_3DPCNet, self).__init__()
        self.encoder_name = encoder_name
        self.decoder_type = decoder_type
        self.face_embedd_size = face_embedd_size
        self.num_output_points = num_output_points
        self.num_axis = num_axis
        self.fp16 = fp16

        if self.encoder_name == "r18":
            self.encoder = iresnet18(False, **kwargs)
        elif self.encoder_name == "r34":
            self.encoder = iresnet34(False, **kwargs)
        elif self.encoder_name == "r50":
            self.encoder = iresnet50(False, **kwargs)
        elif self.encoder_name == "r100":
            self.encoder = iresnet100(False, **kwargs)
        elif self.encoder_name == "r200":
            self.encoder = iresnet200(False, **kwargs)
        elif self.encoder_name == "r2060":
            from .iresnet2060 import iresnet2060
            self.encoder = iresnet2060(False, **kwargs)

        # self.decoder = self.get_decoder_mlp(self.face_embedd_size, self.num_output_points, self.num_axis)
        # self.decoder = self.get_decoder_ConvTranspose2d(input_shape=(1, self.face_embedd_size), output_shape=(self.num_axis, self.num_output_points))
        # self.decoder = self.get_decoder_Conv1x1(input_shape=(1, self.face_embedd_size), output_shape=(1, self.num_axis, self.num_output_points))
        # self.classifier = self.get_classifier(self.num_output_points, self.num_axis, num_classes=2)

        if decoder_type == 'decoder_embedd_multitask1':
            self.decoder = self.get_decoder_embedd(input_shape=(1, self.face_embedd_size), output_shape=(1, self.num_axis, self.num_output_points))

        self.classifier = self.get_classifier(self.num_output_points, self.num_axis, num_classes=2)
        # self._initialize_weights()


    def get_decoder_embedd(self, input_shape=(1, 256), output_shape=(1, 3, 2500)):
        layers = []

        # layer 1 - 0
        conv1_ksize = 3
        layers.append(nn.Conv1d(in_channels=input_shape[0], out_channels=output_shape[2], kernel_size=conv1_ksize, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(output_shape[2], eps=1e-05))
        layers.append(nn.ReLU(True))

        # layer 1 - 1
        conv2_ksize = 5
        layers.append(nn.Conv1d(in_channels=output_shape[2], out_channels=output_shape[2], kernel_size=conv2_ksize, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(output_shape[2], eps=1e-05))
        layers.append(nn.ReLU(True))

        # layer 2
        conv3_ksize = 3
        layers.append(nn.Conv1d(in_channels=output_shape[2], out_channels=output_shape[2], kernel_size=conv3_ksize, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(output_shape[2], eps=1e-05))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def get_decoder_Conv1x1(self, input_shape=(1, 256+(2500*3)), output_shape=(1, 3, 2500)):
        layers = []

        # layer 1 - 0
        conv1_ksize = 258
        pool1_ksize = 29
        layers.append(nn.Conv1d(in_channels=input_shape[0], out_channels=output_shape[2], kernel_size=conv1_ksize, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(output_shape[2], eps=1e-05))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool1d(pool1_ksize))
        # layers.append(nn.AvgPool1d(pool1_ksize))

        # layer 1 - 1
        conv2_ksize = 129
        pool2_ksize = 23
        layers.append(nn.Conv1d(in_channels=output_shape[2], out_channels=129, kernel_size=conv2_ksize, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(129, eps=1e-05))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool1d(pool2_ksize))
        # layers.append(nn.AvgPool1d(pool1_ksize))


        # layer 2
        conv3_ksize = 3
        layers.append(nn.Conv1d(in_channels=129, out_channels=output_shape[2], kernel_size=conv3_ksize, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(output_shape[2], eps=1e-05))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def get_decoder_ConvTranspose2d(self, input_shape=(1, 256), output_shape=(3, 2500)):
        layers = []

        # layer 1
        k_size = (output_shape[1]-input_shape[1]+1, 1)
        layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=129, kernel_size=k_size, stride=1))
        # layers.append(nn.BatchNorm2d(129, eps=1e-05))
        layers.append(nn.ReLU(True))

        # layer 2
        o_channels = output_shape[0]
        layers.append(nn.ConvTranspose2d(in_channels=129, out_channels=o_channels, kernel_size=(1, 1), stride=1))
        # layers.append(nn.BatchNorm2d(output_shape[0], eps=1e-05))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def get_decoder_mlp(self, input_size=256, num_points=2500, num_axis=3):
        layers = []

        # layer 1
        layers.append(nn.Linear(input_size, num_points, bias=False))
        layers.append(nn.BatchNorm1d(num_points, eps=1e-05))
        layers.append(nn.ReLU(True))

        # layer 2
        layers.append(nn.Linear(num_points, num_points*num_axis, bias=False))
        layers.append(nn.BatchNorm1d(num_points*num_axis, eps=1e-05))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def get_classifier(self, num_points=2500, num_axis=3, num_classes=2):
        layers = []

        # layer 1
        layers.append(nn.Linear(num_points*num_axis, num_classes, bias=False))
        layers.append(nn.BatchNorm1d(num_classes, eps=1e-05))

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                # m.weight.data.fill_(1)
                # m.weight.data.fill_(0)
                m.weight.data.uniform_(-1, 1)
                # m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                # m.weight.data.fill_(1)
                # m.weight.data.fill_(0)
                m.weight.data.uniform_(-1, 1)
                # m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, img, pointcloud):
        def _regress_pointcloud_embedd(embedd):                   # input     -> embedd.shape = (batch, 256)
            print('_regress_pointcloud_embedd - embedd.size():', embedd.size())

            embedd = torch.nn.functional.normalize(embedd)
            print('_regress_pointcloud_embedd - embedd.size():', embedd.size())

            embedd = embedd.unsqueeze(1)   # unsqueeze -> embedd_pointcloud.shape = (batch, 1, 7756)
            print('_regress_pointcloud_embedd - embedd.size():', embedd.size())

            pred_pc = self.decoder(embedd)            # decoder   -> pred_pc.shape = (batch, 2500, 3)
            print('_regress_pointcloud_embedd - pred_pc.size():', pred_pc.size())
            sys.exit(0)
            return pred_pc

        def _regress_pointcloud(img, pointcloud):                # input     -> img.shape    = (batch, 3, 224, 224)
            embedd = self.encoder(img)                           # encoder   -> embedd.shape = (batch, 256)
            # print('embedd.size():', embedd.size())
            # print('pointcloud.size():', pointcloud.size())
            pointcloud = pointcloud.reshape(pointcloud.size(0), pointcloud.size(1)*pointcloud.size(2))   # pointcloud.shape = (batch, 2500, 3) -> (batch, 7500)
            # print('reshape - pointcloud.size():', pointcloud.size())

            embedd = torch.nn.functional.normalize(embedd)
            # pointcloud = torch.nn.functional.normalize(pointcloud)
            embedd_pointcloud = torch.cat((embedd, pointcloud), 1)   # embedd_pointcloud.shape = (batch, 7756)
            # print('embedd_pointcloud.size():', embedd_pointcloud.size())

            embedd_pointcloud = embedd_pointcloud.unsqueeze(1)   # unsqueeze -> embedd_pointcloud.shape = (batch, 1, 7756)
            # print('embedd_pointcloud.size():', embedd_pointcloud.size())

            pred_pc = self.decoder(embedd_pointcloud)            # decoder   -> pred_pc.shape = (batch, 2500, 3)
            # print('pred_pc.size():', pred_pc.size())
            # sys.exit(0)
            return pred_pc

        def _get_logits(x):
            x = x.reshape(x.size(0), self.num_output_points*self.num_axis)
            logits = self.classifier(x)
            return logits

        def _forward(img, pointcloud):
            embedd = self.encoder(img)

            if self.decoder_type == 'decoder_embedd_multitask1':
                pred_pc = _regress_pointcloud_embedd(embedd)
                logits = _get_logits(pred_pc)
            return pred_pc, logits

        if self.fp16:
            with torch.cuda.amp.autocast(self.fp16):
                return _forward(img, pointcloud)
        else:
            return _forward(img, pointcloud)



if __name__ == '__main__':
    embedd = torch.zeros((32, 1, 256, 1))
    print('embedd.size():', embedd.size())

    layer1 = nn.Conv2d(in_channels=1, out_channels=2500, kernel_size=(129,1), stride=1, padding=0, bias=False)
    l1_out = layer1(embedd)
    print('l1_out.size():', l1_out.size())