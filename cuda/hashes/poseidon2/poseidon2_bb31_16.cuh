#pragma once

#include "../../fields/bb31_t.cuh"

namespace poseidon2_bb31_16 {

namespace constants {

    constexpr const int DIGEST_WIDTH = 8;
    constexpr const int RATE = 8;
    constexpr const int WIDTH = 16;
    constexpr const int ROUNDS_P = 13;
    constexpr const int ROUNDS_F = 8;
    constexpr const int D = 7;

    __constant__ constexpr const bb31_t INTERNAL_ROUND_CONSTANTS[ROUNDS_P] = {
        bb31_t(1196780786),
        bb31_t(36046858),
        bb31_t(1492041470),
        bb31_t(1864954859),
        bb31_t(883677154),
        bb31_t(1765843422),
        bb31_t(1418914503),
        bb31_t(1404382774),
        bb31_t(859661334),
        bb31_t(1548195514),
        bb31_t(104929687),
        bb31_t(178643863),
        bb31_t(1619872446)
    };

    __constant__ constexpr const bb31_t
        EXTERNAL_ROUND_CONSTANTS[ROUNDS_F * WIDTH] = {
            bb31_t(96748292),   bb31_t(1951698684), bb31_t(177396853),
            bb31_t(719730562),  bb31_t(640767983),  bb31_t(1390633215),
            bb31_t(1716033721), bb31_t(1606702601), bb31_t(1746607367),
            bb31_t(1466015491), bb31_t(1498308946), bb31_t(831109173),
            bb31_t(1029197920), bb31_t(1969905919), bb31_t(83412884),
            bb31_t(1911782445), bb31_t(1693593583), bb31_t(759122502),
            bb31_t(1154399525), bb31_t(1131812921), bb31_t(1080754908),
            bb31_t(53582651),   bb31_t(893583089),  bb31_t(6411452),
            bb31_t(1115338635), bb31_t(580640471),  bb31_t(1264354339),
            bb31_t(842931656),  bb31_t(548879852),  bb31_t(1595288793),
            bb31_t(1562381995), bb31_t(81826002),   bb31_t(262554421),
            bb31_t(1563933798), bb31_t(1440025885), bb31_t(184445025),
            bb31_t(585385439),  bb31_t(1396647410), bb31_t(1575877922),
            bb31_t(1290587480), bb31_t(137125468),  bb31_t(765010148),
            bb31_t(633675867),  bb31_t(24537442),   bb31_t(560123907),
            bb31_t(1895729703), bb31_t(541515871),  bb31_t(1783382863),
            bb31_t(628590563),  bb31_t(1022477421), bb31_t(1659530405),
            bb31_t(245668751),  bb31_t(12194511),   bb31_t(201609705),
            bb31_t(286217151),  bb31_t(66943721),   bb31_t(506306261),
            bb31_t(1067433949), bb31_t(748735911),  bb31_t(1244250808),
            bb31_t(606038199),  bb31_t(1169474910), bb31_t(73007766),
            bb31_t(558938232),  bb31_t(130182653),  bb31_t(742680828),
            bb31_t(542600513),  bb31_t(802837101),  bb31_t(1931786340),
            bb31_t(31204919),   bb31_t(1709908013), bb31_t(925103122),
            bb31_t(1627133772), bb31_t(1374470239), bb31_t(177883755),
            bb31_t(624229761),  bb31_t(209862198),  bb31_t(276092925),
            bb31_t(1820102609), bb31_t(974546524),  bb31_t(1293393192),
            bb31_t(221548340),  bb31_t(1188782305), bb31_t(223782844),
            bb31_t(235714646),  bb31_t(296520220),  bb31_t(10135706),
            bb31_t(1265611492), bb31_t(8872228),    bb31_t(575851471),
            bb31_t(1612560780), bb31_t(1913391015), bb31_t(1305283056),
            bb31_t(578597757),  bb31_t(188109355),  bb31_t(191192067),
            bb31_t(1564209905), bb31_t(140931974),  bb31_t(446421108),
            bb31_t(857368568),  bb31_t(1375012945), bb31_t(1529454825),
            bb31_t(306140690),  bb31_t(842312378),  bb31_t(1246997295),
            bb31_t(1011032842), bb31_t(1915270363), bb31_t(1218245412),
            bb31_t(466048099),  bb31_t(976561834),  bb31_t(814378556),
            bb31_t(13244079),   bb31_t(1165280628), bb31_t(1203983801),
            bb31_t(1801474112), bb31_t(1919627044), bb31_t(600240215),
            bb31_t(773269071),  bb31_t(486685186),  bb31_t(227516968),
            bb31_t(1415023565), bb31_t(502840102),  bb31_t(199116516),
            bb31_t(510217063),  bb31_t(166444818),  bb31_t(1430745893),
            bb31_t(1376516190), bb31_t(1775891321)
    };
#if 1
    __constant__ constexpr const bb31_t MAT_INTERNAL_DIAG_M1[WIDTH] = {
        bb31_t(125829121),  
        bb31_t(943718400),  
        bb31_t(1887436800),
        bb31_t(1761607679),
        bb31_t(1509949437),
        bb31_t(1006632953),
        bb31_t(2013265906),
        bb31_t(2013265891),
        bb31_t(2013265861),
        bb31_t(2013265801),
        bb31_t(2013265681),
        bb31_t(2013265441),
        bb31_t(2013264961),
        bb31_t(2013264001),
        bb31_t(2013262081),
        bb31_t(2013250561)
    };
#else
    __constant__ constexpr const bb31_t MAT_INTERNAL_DIAG_M1[WIDTH] = {
        bb31_t(2013265919),
        bb31_t(1),
        bb31_t(2),
        bb31_t(4),
        bb31_t(8),
        bb31_t(16),
        bb31_t(32),
        bb31_t(64),
        bb31_t(128),
        bb31_t(256),
        bb31_t(512),
        bb31_t(1024),
        bb31_t(2048),
        bb31_t(4096),
        bb31_t(8192),
        bb31_t(32768)
    };
#endif
    __constant__ constexpr const bb31_t MONTY_INVERSE = bb31_t(943718400);

}  // namespace constants

class BabyBear {
  public:
    using F_t = bb31_t;
    using pF_t = const F_t;

    static constexpr const int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr const int RATE = constants::RATE;
    static constexpr const int WIDTH = constants::WIDTH;
    static constexpr const int ROUNDS_F = constants::ROUNDS_F;
    static constexpr const int ROUNDS_P = constants::ROUNDS_P;
    static constexpr const int D = constants::D;

    static constexpr pF_t* INTERNAL_ROUND_CONSTANTS =
        constants::INTERNAL_ROUND_CONSTANTS;
    static constexpr pF_t* EXTERNAL_ROUND_CONSTANTS =
        constants::EXTERNAL_ROUND_CONSTANTS;
    static constexpr pF_t* MAT_INTERNAL_DIAG_M1 =
        constants::MAT_INTERNAL_DIAG_M1;
    static constexpr pF_t MONTY_INVERSE = constants::MONTY_INVERSE;

    __device__ static void internalLinearLayer(F_t state[WIDTH], pF_t*, F_t) {
        uint64_t sum64 = 0;
        for (int i = 0; i < WIDTH; i++) {
            sum64 += static_cast<uint64_t>(state[i].val);
        }
        const F_t sum = bb31_t(static_cast<uint32_t>(sum64 % bb31_t::MOD)) * MONTY_INVERSE;
        for (int i = 0; i < WIDTH; i++) {
            state[i] *= MAT_INTERNAL_DIAG_M1[i];
            state[i] += sum;
        }
    }

    __device__ static void externalLinearLayer(F_t state[WIDTH]) {
        for (int i = 0; i < WIDTH; i += 4) {
            mdsLightPermutation4x4(state + i);
        }
        F_t sums[4] = {state[0], state[1], state[2], state[3]};
        for (int i = 4; i < WIDTH; i += 4) {
            sums[0] += state[i];
            sums[1] += state[i + 1];
            sums[2] += state[i + 2];
            sums[3] += state[i + 3];
        }
        for (int i = 0; i < WIDTH; i++) {
            state[i] += sums[i & 3];
        }
    }

    __device__ static void mdsLightPermutation4x4(F_t state[4]) {
        F_t t01 = state[0] + state[1];
        F_t t23 = state[2] + state[3];
        F_t t0123 = t01 + t23;
        F_t t01123 = t0123 + state[1];
        F_t t01233 = t0123 + state[3];
        state[3] = t01233 + operator<<(state[0], 1);
        state[1] = t01123 + operator<<(state[2], 1);
        state[0] = t01123 + t01;
        state[2] = t01233 + t23;
    }
};

template<typename Hasher_t, typename HasherState_t>
__device__ void absorbRow(
    Hasher_t hasher,
    Matrix<bb31_t>* in,
    int rowIdx,
    HasherState_t* state
) {
    if (in->row_major) {
        bb31_t* row = &in->values[in->width * rowIdx];
        (*state).absorb(hasher, row, in->width);
    } else {
        for (int j = 0; j < in->width; j++) {
            bb31_t* row = &in->values[j * in->height + rowIdx];
            (*state).absorb(hasher, row, 1);
        }
    }
}

}  // namespace poseidon2_bb31_16
