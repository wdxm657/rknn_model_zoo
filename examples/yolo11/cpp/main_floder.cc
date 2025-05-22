// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dirent.h>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103)
#include "dma_alloc.hpp"
#endif

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
// 获取当前时间戳(毫秒)
static int64_t get_current_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// 检查文件是否为图片
static bool is_image_file(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if (ext == NULL)
        return false;

    if (strcasecmp(ext, ".jpg") == 0 ||
        strcasecmp(ext, ".jpeg") == 0 ||
        strcasecmp(ext, ".png") == 0 ||
        strcasecmp(ext, ".bmp") == 0)
    {
        return true;
    }
    return false;
}

static void release_(rknn_app_context_t *rknn_app_ctx)
{
    deinit_post_process();
    int ret;
    if (release_yolo11_model(rknn_app_ctx) != 0)
    {
        printf("release_yolo11_model fail! ret=%d\n", ret);
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_dir>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_dir = argv[2];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        release_(&rknn_app_ctx);
    }

    // 打开目录
    DIR *dir = opendir(image_dir);
    if (dir == NULL)
    {
        printf("Failed to open directory: %s\n", image_dir);
        release_(&rknn_app_ctx);
    }

    struct dirent *entry;
    int total_images = 0;
    int64_t total_time = 0;

    // 遍历目录中的所有文件
    while ((entry = readdir(dir)) != NULL)
    {
        if (!is_image_file(entry->d_name))
        {
            continue;
        }

        // 构建完整的图片路径
        char image_path[256];
        snprintf(image_path, sizeof(image_path), "%s/%s", image_dir, entry->d_name);

        printf("\nProcessing image: %s\n", image_path);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        ret = read_image(image_path, &src_image);
        if (ret != 0)
        {
            printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
            continue;
        }

#if defined(RV1106_1103)
        // RV1106 rga requires that input and output bufs are memory allocated by dma
        ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                            (void **)&(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
        memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
        dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
        free(src_image.virt_addr);
        src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
        src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
        rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif

        object_detect_result_list od_results;
        memset(&od_results, 0, sizeof(object_detect_result_list));

        // 记录开始时间
        int64_t start_time = get_current_time_ms();

        ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolo11_model fail! ret=%d\n", ret);
            if (src_image.virt_addr != NULL)
            {
#if defined(RV1106_1103)
                dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                             rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
                free(src_image.virt_addr);
#endif
            }
            continue;
        }

        // 计算推理时间
        int64_t end_time = get_current_time_ms();
        int64_t inference_time = end_time - start_time;
        total_time += inference_time;
        total_images++;

        // 打印检测结果
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n",
                   coco_cls_to_name(det_result->cls_id),
                   det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom,
                   det_result->prop);
        }

        printf("start time: %ld ms\n", start_time);
        printf("end time: %ld ms\n", end_time);
        printf("Inference time: %ld ms\n", inference_time);

        // 保存结果图片
        char output_path[256];
        snprintf(output_path, sizeof(output_path), "%s/out_%s", image_dir, entry->d_name);

        // 画框和概率
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }

        write_image(output_path, &src_image);

        // 释放图片内存
        if (src_image.virt_addr != NULL)
        {
#if defined(RV1106_1103)
            dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                         rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
            free(src_image.virt_addr);
#endif
        }
    }

    closedir(dir);

    // 打印统计信息
    if (total_images > 0)
    {
        printf("\nTotal images processed: %d\n", total_images);
        printf("Average inference time: %.2f ms\n", (float)total_time / total_images);
    }

    release_(&rknn_app_ctx);

    return 0;
}
