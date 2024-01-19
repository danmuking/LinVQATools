# import torch
import numpy as np
import torch
import torch.nn.functional as F
#
#
import matplotlib.pyplot as plt
from einops import rearrange


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table = torch.tensor(sinusoid_table.reshape(8,7,7,-1))
    sinusoid_table = sinusoid_table.repeat_interleave(2,dim=1).repeat_interleave(2,dim=2)
    sinusoid_table = rearrange(sinusoid_table, "t r w c -> (t r w) c")
    # t = 8
    # h = 14
    # w = 14
    # sinusoid_table = rearrange(sinusoid_table, "(t r w) c -> t r w c", t=t, r=h, w=w)
    # sinusoid_table = rearrange(sinusoid_table, 't r (w c1) c -> (t r w) c1 c', w=w // 2, c1=2)
    # sinusoid_table = rearrange(sinusoid_table, '(n r) w c -> n r w c', n=t * h * w // 2 // 4, r=4)
    # sinusoid_table = rearrange(sinusoid_table, '(n1 n2 n3) (a b) w c -> n1 n2 n3 a b w c', n1=t // 2, n2=h // 2,
    #                            n3=w // 2, a=2, b=2)
    # sinusoid_table = rearrange(sinusoid_table, 'n1 n2 n3 a b w c -> (n1 a) (n2 b) (n3 w) c')
    # sinusoid_table = rearrange(sinusoid_table, 'n1 n2 n3 c -> (n1 n2 n3) c')

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

def get_sinusoid_encoding_table1(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    t = 8
    h = 14
    w = 14
    sinusoid_table = rearrange(sinusoid_table, "(t r w) c -> t r w c", t=t, r=h, w=w)
    sinusoid_table = rearrange(sinusoid_table, 't r (w c1) c -> (t r w) c1 c', w=w // 2, c1=2)
    sinusoid_table = rearrange(sinusoid_table, '(n r) w c -> n r w c', n=t * h * w // 2 // 4, r=4)
    sinusoid_table = rearrange(sinusoid_table, '(n1 n2 n3) (a b) w c -> n1 n2 n3 a b w c', n1=t // 2, n2=h // 2,
                               n3=w // 2, a=2, b=2)
    sinusoid_table = rearrange(sinusoid_table, 'n1 n2 n3 a b w c -> (n1 a) (n2 b) (n3 w) c')
    sinusoid_table = rearrange(sinusoid_table, 'n1 n2 n3 c -> (n1 n2 n3) c')

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

if __name__ == '__main__':
    # pos_embed = get_sinusoid_encoding_table(7 * 7 * 8, 384)
    # print(pos_embed.shape)
    # pos_embed = pos_embed[0]
    # print(pos_embed[0, :].unsqueeze(0).repeat(14 * 14 * 8, 1).shape)
    # print(pos_embed.shape)
    #
    # similar = F.cosine_similarity(pos_embed[1, :].unsqueeze(0).repeat(14 * 14 * 8, 1), pos_embed).reshape(8, 14, 14)
    # print(similar.shape)
    # # print(similar)
    #
    # fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    # plt.subplots_adjust(wspace=0.2, hspace=0.0)  # 调整子图间距
    # img = None
    # for i in range(8):
    #     ax = axs[0][i]
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_frame_on(False)
    #     img = ax.imshow(similar[i].reshape(14, 14), cmap=plt.cm.Reds)
    #
    # pos_embed = get_sinusoid_encoding_table1(14 * 14 * 8, 384)
    # print(pos_embed.shape)
    # pos_embed = pos_embed[0]
    # print(pos_embed[0, :].unsqueeze(0).repeat(14 * 14 * 8, 1).shape)
    # print(pos_embed.shape)
    #
    # similar = F.cosine_similarity(pos_embed[1, :].unsqueeze(0).repeat(14 * 14 * 8, 1), pos_embed).reshape(8, 14, 14)
    # print(similar.shape)
    # for i in range(8):
    #     ax = axs[1][i]
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_frame_on(False)
    #     img = ax.imshow(similar[i].reshape(14, 14), cmap=plt.cm.Reds)
    # cbar = fig.colorbar(img, ax=axs, orientation='vertical', pad=0.03, shrink=0.83)
    # cbar.set_label('Cosine Similarity', size=12)
    # cbar.set_ticks([cbar.vmin, cbar.vmax])
    # cbar.set_ticklabels(['low', 'high'])
    # # 为最下面的子图添加标签
    # for j, ax in enumerate(axs[1]):
    #     ax.text(0.5, -0.2, f"{j + 1}", ha='center', va='center', transform=ax.transAxes)
    #
    # # 在每一行的左侧添加小段文字
    # left_edge = axs[0][0].get_position().get_points()[0][0]
    # fig.text(left_edge - 0.01,
    #          (axs[0][0].get_position().get_points()[0][1] + axs[0][1].get_position().get_points()[1][1]) / 2,
    #          f'Sinusoidal', ha='center', va='center', rotation='vertical', fontsize=12)
    # left_edge = axs[1][0].get_position().get_points()[0][0]
    # fig.text(left_edge - 0.01,
    #          (axs[1][0].get_position().get_points()[0][1] + axs[1][1].get_position().get_points()[1][1]) / 2,
    #          f'Ours', ha='center', va='center', rotation='vertical', fontsize=12)
    #
    # fig.text(0.444, 0.05, 'Input frame', ha='center', va='center', fontsize=16)
    # # 添加大标题（手动创建辅助子图）
    # ax_title = fig.add_subplot(111, frame_on=False)
    # ax_title.set_xticks([])
    # ax_title.set_yticks([])
    # ax_title.set_frame_on(False)
    # ax_title.text(0.444, 1.05, 'Position embedding similarity', ha='center', va='center', fontsize=16,
    #               fontweight='bold')
    #
    # plt.show()
    #

    # fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    # plt.subplots_adjust(wspace=0.2, hspace=-0.6)  # 调整子图间距

    # pos_embed = get_sinusoid_encoding_table1(14 * 14 * 8, 384)
    # pos_embed = pos_embed[0]
    # for j in range(0*8,1*8):
    #     similar = []
    #     for i in range(pos_embed.shape[0]):
    #         similar.append(F.cosine_similarity(pos_embed[j].unsqueeze(0),pos_embed[i].unsqueeze(0)))
    #     similar = torch.tensor(similar)
    #     # print(similar)
    #     # similar = rearrange(similar, '(c1 t) (c2 w) (c3 h) -> (c1 c2 c3) t w h', c1=4, c2=7, c3=7, t=2, w=2, h=2)
    #     # print(similar[0])
    #     x1 = similar[0*8:1*8]
    #     x2 = similar[10*8:11*8]
    #     x = torch.stack([x1,x2]).reshape(2,-1)
    #     ax = axs[0][j]
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_frame_on(False)
    #     img = ax.imshow(x)

    # for j in range(0,8):
    #     similar = []
    #     for i in range(pos_embed.shape[0]):
    #         similar.append(F.cosine_similarity(pos_embed[j+80].unsqueeze(0),pos_embed[i].unsqueeze(0)))
    #     similar = torch.tensor(similar)
    #     # print(similar)
    #     # similar = rearrange(similar, '(c1 t) (c2 w) (c3 h) -> (c1 c2 c3) t w h', c1=4, c2=7, c3=7, t=2, w=2, h=2)
    #     # print(similar[0])
    #     x1 = similar[0*8:1*8]
    #     x2 = similar[10*8:11*8]
    #     x = torch.stack([x1,x2]).reshape(2,-1)
    #     ax = axs[1][j]
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_frame_on(False)
    #     img = ax.imshow(x)
    # cbar = fig.colorbar(img, ax=axs, orientation='vertical', pad=0.03, shrink=0.83,fraction=0.006)
    # cbar.set_label('Cosine Similarity', size=12)
    # cbar.set_ticks([cbar.vmin, cbar.vmax])
    # cbar.set_ticklabels(['low', 'high'])
    # # # 为最下面的子图添加标签
    # # for j, ax in enumerate(axs[1]):
    # #     ax.text(0.5, -0.2, f"{j + 1}", ha='center', va='center', transform=ax.transAxes)
    #
    # # 在每一行的左侧添加小段文字
    # left_edge = axs[0][0].get_position().get_points()[0][0]
    # # fig.text(left_edge - 0.01,
    # #          (axs[0][0].get_position().get_points()[0][1] + axs[0][1].get_position().get_points()[1][1]) / 2,
    # #          f'Sinusoidal', ha='center', va='center', fontsize=12)
    # # left_edge = axs[1][0].get_position().get_points()[0][0]
    # # fig.text(left_edge - 0.01,
    # #          (axs[1][0].get_position().get_points()[0][1] + axs[1][1].get_position().get_points()[1][1]) / 2,
    # #          f'Ours', ha='center', va='center', fontsize=12)
    #
    # fig.text(0.444, 0.25, 'Position embedding similarity', ha='center', va='center', fontsize=18)
    # # 添加大标题（手动创建辅助子图）
    # # ax_title = fig.add_subplot(111, frame_on=False)
    # # ax_title.set_xticks([])
    # # ax_title.set_yticks([])
    # # ax_title.set_frame_on(False)
    # # ax_title.text(0.444, 1.05, 'Position embedding similarity', ha='center', va='center', fontsize=16,
    # #               fontweight='bold')
    # #
    # plt.show()

    similar = []
    pos_embed = get_sinusoid_encoding_table1(14 * 14 * 8, 384)
    pos_embed = pos_embed[0]
    pos_embed = pos_embed[14*8:14*8+8]
    for i in range(pos_embed.shape[0]):
        for j in range(pos_embed.shape[0]):
            similar.append(F.cosine_similarity(pos_embed[i].unsqueeze(0), pos_embed[j].unsqueeze(0)))
    similar = torch.tensor(similar).reshape(8,8)
    plt.axis('off')  # 取消坐标轴
    plt.imshow(similar,vmin=0.2690, vmax=1)
    plt.show()

    similar = []
    pos_embed = get_sinusoid_encoding_table1(14 * 14 * 8, 384)
    pos_embed = pos_embed[0]
    diff_pos_embed = pos_embed[7*8:7*8+8]
    pos_embed = pos_embed[:8]
    for i in range(pos_embed.shape[0]):
        for j in range(pos_embed.shape[0]):
            similar.append(F.cosine_similarity(pos_embed[i].unsqueeze(0), diff_pos_embed[j].unsqueeze(0)))
    similar = torch.tensor(similar).reshape(8, 8)
    print(similar.min())
    plt.axis('off')  # 取消坐标轴
    plt.imshow(similar, vmin=0.2690, vmax=1)
    plt.show()

    similar = []
    pos_embed = get_sinusoid_encoding_table1(14 * 14 * 8, 384)
    pos_embed = pos_embed[0]
    diff_pos_embed = pos_embed[15*8:15*8 + 8]
    pos_embed = pos_embed[:8]
    for i in range(pos_embed.shape[0]):
        for j in range(pos_embed.shape[0]):
            similar.append(F.cosine_similarity(pos_embed[i].unsqueeze(0), diff_pos_embed[j].unsqueeze(0)))
    similar = torch.tensor(similar).reshape(8, 8)
    print(similar.min())
    plt.axis('off')  # 取消坐标轴
    plt.imshow(similar, vmin=0.2690, vmax=1)
    plt.show()

    similar = []
    pos_embed = get_sinusoid_encoding_table1(14 * 14 * 8, 384)
    pos_embed = pos_embed[0]
    diff_pos_embed = pos_embed[21*8:21*8 + 8]
    pos_embed = pos_embed[:8]
    for i in range(pos_embed.shape[0]):
        for j in range(pos_embed.shape[0]):
            similar.append(F.cosine_similarity(pos_embed[i].unsqueeze(0), diff_pos_embed[j].unsqueeze(0)))
    similar = torch.tensor(similar).reshape(8, 8)
    print(similar.min())
    plt.axis('off')  # 取消坐标轴
    plt.imshow(similar, vmin=0.2690, vmax=1)
    cbar = plt.colorbar()  # 添加colorbar
    cbar.set_label('Cosine Similarity', size=12)
    cbar.set_ticks([cbar.vmin, cbar.vmax])
    cbar.set_ticklabels(['low', 'high'])
    plt.show()