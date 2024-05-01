# Copyright 2024 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# tif2jpg.py

import cv2
import os
import glob
import shutil

import traceback


def tif2jpg(test_images_dir, output_images_dir):
  if not os.path.exists(output_images_dir):
     os.makedirs(output_images_dir)

  tif_files = glob.glob(test_images_dir + "/*.tif")
  for tif_file in tif_files:
    image = cv2.imread(tif_file)
    basename = os.path.basename(tif_file)
    basename = basename.replace(".tif", ".jpg")
    output_file = os.path.join(output_images_dir, basename)
    cv2.imwrite(output_file, image)
    print("--- Saved {}".format(output_file))

if __name__ == "__main__":
  try:
    test_images_dir = "./mini_test_tif/images"
    test_masks_dir  = "./mini_test_tif/masks"

    output_dir           = "./mini_test"
 
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   
    output_images_dir  = "./mini_test/images"
    output_masks_dir   = "./mini_test/masks"

    tif2jpg(test_images_dir, output_images_dir)
    tif2jpg(test_masks_dir,  output_masks_dir)


  except:
    traceback.print_exc()
