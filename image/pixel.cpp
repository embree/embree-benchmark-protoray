// ======================================================================== //
// Copyright 2015-2017 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "pixel.h"

namespace prt {

/*
for (int i = 0; i < 256; ++i)
{
    float c = pow((float)i / 255.0f, 2.2f);

    if (i % 4 == 0)
        printf("\n    ");
    else
        printf(" ");

    printf("%14a,", c);
}
*/
ALIGNED_CACHE const float decodeSrgb8Table[256] =
{
            0x0p+0, 0x1.54b714p-18, 0x1.876102p-16, 0x1.dd7e8ap-15,
    0x1.c1938cp-14, 0x1.6f42f6p-13, 0x1.123f9cp-12,  0x1.80f86p-12,
    0x1.0236bcp-11, 0x1.4e9772p-11, 0x1.a5df6ep-11, 0x1.04251cp-10,
    0x1.3b075ep-10, 0x1.77b006p-10, 0x1.ba36f4p-10,  0x1.01594ep-9,
     0x1.289c1cp-9,  0x1.52ede8p-9,  0x1.805846p-9,  0x1.b0e452p-9,
     0x1.e49ac4p-9,  0x1.0dc1fep-8,  0x1.2ad3fep-8,   0x1.49874p-8,
      0x1.69df8p-8,  0x1.8be056p-8,  0x1.af8d3ep-8,  0x1.d4e998p-8,
     0x1.fbf8a6p-8,  0x1.125eccp-7,   0x1.279dcp-7,  0x1.3dbab2p-7,
     0x1.54b718p-7,  0x1.6c9458p-7,  0x1.8553dcp-7,  0x1.9ef6fcp-7,
     0x1.b97f0cp-7,  0x1.d4ed56p-7,   0x1.f1432p-7,  0x1.0740d2p-6,
     0x1.16550ep-6,  0x1.25dedap-6,  0x1.35deccp-6,  0x1.465578p-6,
     0x1.57436ap-6,   0x1.68a93p-6,  0x1.7a8756p-6,  0x1.8cde62p-6,
     0x1.9faed8p-6,  0x1.b2f93ep-6,  0x1.c6be14p-6,  0x1.dafdd8p-6,
     0x1.efb908p-6,   0x1.02781p-5,  0x1.0d51cap-5,  0x1.1869f2p-5,
     0x1.23c0bep-5,  0x1.2f566cp-5,  0x1.3b2b32p-5,  0x1.473f48p-5,
     0x1.5392eap-5,  0x1.60264ap-5,   0x1.6cf9ap-5,  0x1.7a0d22p-5,
     0x1.876106p-5,  0x1.94f57cp-5,  0x1.a2cab8p-5,  0x1.b0e0eep-5,
      0x1.bf385p-5,  0x1.cdd10ep-5,  0x1.dcab5cp-5,  0x1.ebc76ap-5,
     0x1.fb2564p-5,  0x1.0562bep-4,  0x1.0d53f2p-4,  0x1.156662p-4,
     0x1.1d9a26p-4,  0x1.25ef56p-4,  0x1.2e6606p-4,   0x1.36fe5p-4,
     0x1.3fb844p-4,  0x1.4893fep-4,   0x1.51919p-4,  0x1.5ab10ep-4,
     0x1.63f292p-4,  0x1.6d562cp-4,  0x1.76dbf4p-4,  0x1.8083fcp-4,
     0x1.8a4e58p-4,   0x1.943b2p-4,  0x1.9e4a64p-4,  0x1.a87c3ap-4,
     0x1.b2d0b4p-4,  0x1.bd47e8p-4,  0x1.c7e1e6p-4,  0x1.d29ec2p-4,
      0x1.dd7e9p-4,  0x1.e88162p-4,  0x1.f3a74ap-4,  0x1.fef05cp-4,
     0x1.052e54p-3,  0x1.0af622p-3,   0x1.10cfap-3,  0x1.16bad6p-3,
     0x1.1cb7cep-3,   0x1.22c69p-3,  0x1.28e726p-3,  0x1.2f1996p-3,
     0x1.355decp-3,  0x1.3bb42ep-3,  0x1.421c66p-3,  0x1.48969cp-3,
     0x1.4f22d6p-3,   0x1.55c12p-3,  0x1.5c7182p-3,    0x1.6334p-3,
     0x1.6a08a8p-3,  0x1.70ef7cp-3,  0x1.77e888p-3,  0x1.7ef3d4p-3,
     0x1.861166p-3,  0x1.8d4146p-3,  0x1.94837ep-3,  0x1.9bd814p-3,
     0x1.a33f0ep-3,  0x1.aab876p-3,  0x1.b24454p-3,  0x1.b9e2aep-3,
      0x1.c1939p-3,  0x1.c956fap-3,  0x1.d12cf8p-3,  0x1.d9158ep-3,
     0x1.e110c6p-3,  0x1.e91ea8p-3,  0x1.f13f3ap-3,  0x1.f97282p-3,
     0x1.00dc44p-2,  0x1.0508aap-2,  0x1.093e78p-2,  0x1.0d7daep-2,
     0x1.11c652p-2,  0x1.161866p-2,  0x1.1a73eep-2,  0x1.1ed8eep-2,
     0x1.234768p-2,  0x1.27bf62p-2,  0x1.2c40dcp-2,  0x1.30cbdcp-2,
     0x1.356064p-2,  0x1.39fe78p-2,  0x1.3ea61cp-2,   0x1.43575p-2,
     0x1.48121cp-2,  0x1.4cd67ep-2,  0x1.51a47ep-2,  0x1.567c1cp-2,
     0x1.5b5d5ep-2,  0x1.604844p-2,  0x1.653cd4p-2,   0x1.6a3b1p-2,
     0x1.6f42fap-2,  0x1.745496p-2,  0x1.796fe8p-2,  0x1.7e94f2p-2,
     0x1.83c3b6p-2,  0x1.88fc3ap-2,  0x1.8e3e7ep-2,  0x1.938a88p-2,
     0x1.98e05ap-2,  0x1.9e3ff4p-2,  0x1.a3a95ep-2,  0x1.a91c96p-2,
     0x1.ae99a2p-2,  0x1.b42086p-2,   0x1.b9b14p-2,  0x1.bf4bd8p-2,
      0x1.c4f05p-2,  0x1.ca9ea8p-2,  0x1.d056e6p-2,  0x1.d6190cp-2,
     0x1.dbe51ap-2,  0x1.e1bb18p-2,  0x1.e79b04p-2,  0x1.ed84e4p-2,
     0x1.f378bap-2,  0x1.f97688p-2,   0x1.ff7e5p-2,  0x1.02c80cp-1,
      0x1.05d5fp-1,  0x1.08e8d6p-1,  0x1.0c00bep-1,  0x1.0f1dacp-1,
     0x1.123f9ep-1,  0x1.156698p-1,  0x1.18929cp-1,  0x1.1bc3a8p-1,
      0x1.1ef9cp-1,  0x1.2234e4p-1,  0x1.257514p-1,  0x1.28ba56p-1,
     0x1.2c04a8p-1,  0x1.2f540ap-1,   0x1.32a88p-1,  0x1.36020ap-1,
     0x1.3960aap-1,   0x1.3cc46p-1,   0x1.402d3p-1,  0x1.439b18p-1,
     0x1.470e1ap-1,  0x1.4a8638p-1,  0x1.4e0374p-1,   0x1.5185dp-1,
     0x1.550d48p-1,  0x1.5899e4p-1,   0x1.5c2bap-1,  0x1.5fc282p-1,
     0x1.635e86p-1,   0x1.66ffbp-1,  0x1.6aa602p-1,  0x1.6e517cp-1,
      0x1.72022p-1,  0x1.75b7eep-1,  0x1.7972e8p-1,   0x1.7d331p-1,
     0x1.80f864p-1,  0x1.84c2eap-1,  0x1.88929ep-1,  0x1.8c6786p-1,
      0x1.9041ap-1,  0x1.9420eep-1,  0x1.980572p-1,  0x1.9bef2cp-1,
     0x1.9fde1ep-1,  0x1.a3d248p-1,  0x1.a7cbaep-1,  0x1.abca4cp-1,
     0x1.afce28p-1,  0x1.b3d742p-1,  0x1.b7e59ap-1,   0x1.bbf93p-1,
     0x1.c01208p-1,  0x1.c43022p-1,   0x1.c8538p-1,   0x1.cc7c2p-1,
     0x1.d0aa06p-1,  0x1.d4dd34p-1,  0x1.d915a8p-1,  0x1.dd5364p-1,
     0x1.e1966ap-1,  0x1.e5debcp-1,  0x1.ea2c58p-1,  0x1.ee7f42p-1,
     0x1.f2d77ap-1,    0x1.f735p-1,  0x1.fb97d8p-1,         0x1p+0,
};

} // namespace prt
