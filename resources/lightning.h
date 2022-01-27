#ifndef LIGHTNING_H
#define LIGHTNING_H

#include <cassert>
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include <cmath>

#define LIGHTNING_DEVICE __device__ __forceinline__
#define LIGHTNING_HOST_DEVICE __host__ __device__ __forceinline__
#define LIGHTNING_HOST_DEVICE_NOINLINE __host__ __device__ __noinline__

#ifndef LIGHTNING_BOUNDS_CHECKING
#if NDEBUG
#define LIGHTNING_BOUNDS_CHECKING (0)
#else
#define LIGHTNING_BOUNDS_CHECKING (1)
#endif
#endif

#if NDEBUG
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 2
#define LIGHTNING_ASSUME(expr) __builtin_assume(expr)
#else
#define LIGHTNING_ASSUME(expr) do {} while (false)
#endif
#else
#define LIGHTNING_ASSUME(expr) assert(expr)
#endif

namespace lightning {

namespace reductions {
    template <typename T>
    LIGHTNING_HOST_DEVICE T sum(T lhs, T rhs) {
        return lhs + rhs;
    }

    template <typename T>
    LIGHTNING_HOST_DEVICE T product(T lhs, T rhs) {
        return lhs * rhs;
    }

    template <typename T>
    LIGHTNING_HOST_DEVICE T bit_and(T lhs, T rhs) {
        return lhs & rhs;
    }

    template <typename T>
    LIGHTNING_HOST_DEVICE T bit_or(T lhs, T rhs) {
        return lhs | rhs;
    }

    template <typename T>
    LIGHTNING_HOST_DEVICE T min(T lhs, T rhs) {
        return lhs < rhs ? lhs : rhs;
    }

    template <typename T>
    LIGHTNING_HOST_DEVICE T max(T lhs, T rhs) {
        return lhs > rhs ? lhs : rhs;
    }
}

template <typename K, typename V>
struct KeyValuePair {
    public:
        LIGHTNING_HOST_DEVICE KeyValuePair(): key(), value() {
            //
        }

        LIGHTNING_HOST_DEVICE KeyValuePair(K key, V value): key(key), value(value) {
            //
        }

        LIGHTNING_HOST_DEVICE bool operator==(const KeyValuePair &other) const {
            return key == other.key && value == other.value;
        }

        LIGHTNING_HOST_DEVICE bool operator!=(const KeyValuePair &other) const {
            return !(*this == other);
        }

        LIGHTNING_HOST_DEVICE bool operator<(const KeyValuePair &other) const {
            return value < other.value;
        }

        LIGHTNING_HOST_DEVICE bool operator>(const KeyValuePair &other) const {
            return value > other.value;
        }

        LIGHTNING_HOST_DEVICE bool operator<=(const KeyValuePair &other) const {
            return value <= other.value;
        }

        LIGHTNING_HOST_DEVICE bool operator>=(const KeyValuePair &other) const {
            return value >= other.value;
        }

        K key;
        V value;
};

template <typename T> using IndexValuePair = KeyValuePair<int64_t, T>;

template <typename T, size_t N>
struct Tuple {
public:
    LIGHTNING_HOST_DEVICE const T& operator[](const size_t i) const {
        return m_inner[i];
    }

    LIGHTNING_HOST_DEVICE T& operator[](const size_t i) {
        return m_inner[i];
    }

    template <size_t Index>
    LIGHTNING_HOST_DEVICE Tuple<T, N - 1> drop_element() const {
        static_assert(Index < N);
        Tuple<T, N - 1> output;

#pragma unroll
        // TODO: compiler complains about "pointless comparison of unsigned integer with zero", find workaround
        for (size_t i = 0; (int)i < (int)Index; i++) {
            output[i] = m_inner[i];
        }

#pragma unroll
        for (size_t i = Index; (int)i < (int)N - 1; i++) {
            output[i] = m_inner[i + 1];
        }

        return output;
    }

    LIGHTNING_HOST_DEVICE static Tuple repeat(const T value) {
        Tuple output;

        for (size_t i = 0; i < N; i++) {
            output[i] = value;
        }

        return output;
    }

    //private:
    T m_inner[N];
};

template <typename T>
struct Tuple<T, 0> {
public:
    LIGHTNING_HOST_DEVICE const T& operator[](const size_t i) const {
        while (true);
    }

    LIGHTNING_HOST_DEVICE T& operator[](const size_t i) {
        while (true);
    }

    LIGHTNING_HOST_DEVICE static Tuple repeat(const T value) {
        return {};
    }
};

template <size_t N> using Point = Tuple<int64_t, N>;
template <size_t N> using Strides = Tuple<int64_t, N>;

template <size_t N>
LIGHTNING_HOST_DEVICE_NOINLINE void panic_out_of_bounds(Point<N> indices, Point<N> lbnd, Point<N> ubnd);

template <size_t N, bool BoundsCheck>
struct ArrayBounds {
    LIGHTNING_HOST_DEVICE ArrayBounds(const Point<N> lbnd, const Point<N> ubnd) {
        //
    }

    LIGHTNING_HOST_DEVICE ArrayBounds() {
        //
    }

    LIGHTNING_HOST_DEVICE void check_indices_bounds(const Point<N> indices) const {
        //
    }

    template <size_t Axis>
    LIGHTNING_HOST_DEVICE void check_index_bounds(int64_t index) const {
        //
    }

    LIGHTNING_HOST_DEVICE const Point<N> lbnd() const {
        return Point<N>::repeat(0);
    }

    LIGHTNING_HOST_DEVICE const Point<N> ubnd() const {
        return Point<N>::repeat(INT64_MAX);
    }
};

template <size_t N>
struct ArrayBounds<N, true> {
    LIGHTNING_HOST_DEVICE ArrayBounds(const Point<N> lbnd, const Point<N> ubnd) noexcept:
        m_lbnd(lbnd),
        m_ubnd(ubnd)
    {
        //
    }

    LIGHTNING_HOST_DEVICE ArrayBounds():
        m_lbnd(Point<N>::repeat(0)),
        m_ubnd(Point<N>::repeat(INT64_MAX))
    {
        //
    }

    LIGHTNING_HOST_DEVICE void check_indices_bounds(const Point<N> indices) {
        bool out_bounds = false;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            out_bounds = out_bounds | (indices[i] < m_lbnd[i]);
            out_bounds = out_bounds | (indices[i] >= m_ubnd[i]);
        }

        if (__builtin_expect(out_bounds, false)) {
            panic_out_of_bounds(indices, m_lbnd, m_ubnd); // will interrupt kernel execution.
            while (1); // helps compiler notice function can never return beyond this point.
        }
    }

    template <size_t Axis>
    LIGHTNING_HOST_DEVICE void check_index_bounds(int64_t index) const {
        static_assert(Axis < N, "Axis of out bounds");
        bool out_bounds = (index < m_lbnd[Axis]) | (index >= m_ubnd[Axis]);

        if (__builtin_expect(out_bounds, false)) {
            panic_out_of_bounds<1>({index}, {m_lbnd[Axis]}, {m_ubnd[Axis]}); // will interrupt kernel execution.
            while (1); // helps compiler notice function can never return beyond this point.
        }
    }

    LIGHTNING_HOST_DEVICE const Point<N> lbnd() const {
        return m_lbnd;
    }

    LIGHTNING_HOST_DEVICE const Point<N> ubnd() const {
        return m_ubnd;
    }

    private:
        const Point<N> m_lbnd;
        const Point<N> m_ubnd;
};

template <typename T, size_t N, bool BoundsCheck>
struct Array;

template <typename T, size_t N, bool BoundsCheck>
struct ArrayBase: ArrayBounds<N, BoundsCheck> {
    public:
    LIGHTNING_HOST_DEVICE ArrayBase(T *const ptr, const Strides<N> strides, const Point<N> lbnd, const Point<N> ubnd) noexcept:
            m_base(ptr),
            m_strides(strides),
            ArrayBounds<N, BoundsCheck>(lbnd, ubnd)
    {
        //
    }

    LIGHTNING_HOST_DEVICE ArrayBase(T *const ptr, const Strides<N> strides, const Point<N> ubnd) noexcept:
            m_base(ptr),
            m_strides(strides),
            ArrayBounds<N, BoundsCheck>(Point<N>::repeat(0), ubnd)
    {
        //
    }

    LIGHTNING_HOST_DEVICE ArrayBase(T *const ptr, const Strides<N> strides) noexcept:
            m_base(ptr),
            m_strides(strides),
            ArrayBounds<N, BoundsCheck>()
    {
        //
    }

    LIGHTNING_HOST_DEVICE const T& get(const Point<N> indices) const {
        this->check_indices_bounds(indices);

        T *ptr = m_base;

        #pragma unroll
        for (size_t i = 0; i < N; i++) {
            ptr = ptr + indices[i] * m_strides[i];
        }

        return *ptr;
    }

    LIGHTNING_HOST_DEVICE T& get(const Point<N> indices) {
        this->check_indices_bounds(indices);

        T *ptr = m_base;

        #pragma unroll
        for (size_t i = 0; i < N; i++) {
            ptr = ptr + indices[i] * m_strides[i];
        }

        return *ptr;
    }

    template <size_t Axis>
    LIGHTNING_HOST_DEVICE Array<T, N - 1, BoundsCheck> collapse_axis(int64_t index) {
        static_assert(Axis < N, "Axis of out bounds");
        this->template check_index_bounds<Axis>(index);

        T *new_ptr = m_base + index * m_strides[Axis];

        Point<N - 1> new_lbnd = this->lbnd().template drop_element<Axis>();
        Point<N - 1> new_ubnd = this->ubnd().template drop_element<Axis>();
        Point<N - 1> new_strides = m_strides.template drop_element<Axis>();

        return Array<T, N - 1, BoundsCheck>((T*) new_ptr, new_strides, new_lbnd, new_ubnd);
    }

    template <size_t Axis>
    LIGHTNING_HOST_DEVICE const Array<T, N - 1, BoundsCheck> collapse_axis(int64_t index) const {
        static_assert(Axis < N, "Axis of out bounds");
        this->template check_index_bounds<Axis>(index);

        T *new_ptr = m_base + index * m_strides[Axis];

        Point<N - 1> new_lbnd = this->lbnd().template drop_element<Axis>();
        Point<N - 1> new_ubnd = this->ubnd().template drop_element<Axis>();
        Point<N - 1> new_strides = m_strides.template drop_element<Axis>();

        return Array<T, N - 1, BoundsCheck>((T*) new_ptr, new_strides, new_lbnd, new_ubnd);
    }

    LIGHTNING_HOST_DEVICE const Array<T, N - 1, BoundsCheck> operator[](const int64_t i) const {
        return this->template collapse_axis<0>(i);
    }

    LIGHTNING_HOST_DEVICE Array<T, N - 1, BoundsCheck> operator[](const int64_t i) {
        return this->template collapse_axis<0>(i);
    }

    //private:
        T *const m_base;
        const Strides<N> m_strides;
};

// specialization for N=0, no need to store anything except the data ptr.
template <typename T, bool BoundsCheck>
struct ArrayBase<T, 0, BoundsCheck>: ArrayBounds<0, BoundsCheck> {
    public:
        LIGHTNING_HOST_DEVICE ArrayBase(T *const ptr) noexcept:
            m_base(ptr) {
            //
        }

        LIGHTNING_HOST_DEVICE const T& get(const Point<0> indices) const noexcept {
            return *m_base;
        }

        LIGHTNING_HOST_DEVICE T& get(const Point<0> indices) noexcept {
            return *m_base;
        }

    //private:
        T *const m_base;
};


template <typename T, size_t N, bool BoundsCheck = LIGHTNING_BOUNDS_CHECKING>
struct Array: ArrayBase<T, N, BoundsCheck> {
    LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<N> strides, const Point<N> lbnd, const Point<N> ubnd) noexcept:
        ArrayBase<T, N, BoundsCheck>(ptr, strides, lbnd, ubnd) {
        //
    }

    LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<N> strides, const Point<N> ubnd) noexcept:
        ArrayBase<T, N, BoundsCheck>(ptr, strides, ubnd) {
        //
    }

    LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<N> strides) noexcept:
        ArrayBase<T, N, BoundsCheck>(ptr, strides) {
        //
    }
};

template <typename T, bool BoundsCheck>
struct Array<T, 0, BoundsCheck>: ArrayBase<T, 0, BoundsCheck> {
    public:
        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<0> strides, const Point<0> lbnd, const Point<0> ubnd) noexcept:
            ArrayBase<T, 0, BoundsCheck>(ptr) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<0> strides, const Point<0> ubnd) noexcept:
            ArrayBase<T, 0, BoundsCheck>(ptr) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<0> strides) noexcept:
            ArrayBase<T, 0, BoundsCheck>(ptr) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr) noexcept:
            ArrayBase<T, 0, BoundsCheck>(ptr, {}, {}, {}) {
            //
        }

        LIGHTNING_HOST_DEVICE const T& operator()() const {
            return this->get({});
        }

        LIGHTNING_HOST_DEVICE T& operator()() {
            return this->get({});
        }

        LIGHTNING_HOST_DEVICE const T& operator*() const {
            return this->get({});
        }

        LIGHTNING_HOST_DEVICE T& operator*() {
            return this->get({});
        }

        LIGHTNING_HOST_DEVICE operator T&() {
            return this->get({});
        }

        LIGHTNING_HOST_DEVICE operator const T&() const {
            return this->get({});
        }

        LIGHTNING_HOST_DEVICE Array& operator=(T value) noexcept {
            this->get({}) = value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE Array& operator+=(T value) noexcept {
            this->get({}) += value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE Array& operator-=(T value) noexcept {
            this->get({}) -= value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE Array& operator*=(T value) noexcept {
            this->get({}) *= value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE Array& operator/=(T value) noexcept {
            this->get({}) /= value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE Array& operator|=(T value) noexcept {
            this->get({}) |= value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE Array& operator&=(T value) noexcept {
            this->get({}) &= value;
            return *this;
        }

        LIGHTNING_HOST_DEVICE T* operator&() noexcept {
            return &this->get({});
        }

        LIGHTNING_HOST_DEVICE const T* operator&() const noexcept {
            return &this->get({});
        }
};

template <typename T, bool BoundsCheck>
struct Array<T, 1, BoundsCheck>: ArrayBase<T, 1, BoundsCheck> {
    public:
        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<1> strides, const Point<1> lbnd, const Point<1> ubnd) noexcept:
            ArrayBase<T, 1, BoundsCheck>(ptr, strides, lbnd, ubnd) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<1> strides, const Point<1> ubnd) noexcept:
            ArrayBase<T, 1, BoundsCheck>(ptr, strides, ubnd) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<1> strides) noexcept:
            ArrayBase<T, 1, BoundsCheck>(ptr, strides) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, int64_t size) noexcept:
            Array(ptr, 0, size, 1) {

        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, int64_t lbnd, int64_t ubnd, int64_t stride=1) noexcept:
            ArrayBase<T, 1, BoundsCheck>(ptr, {lbnd}, {ubnd}, {stride}) {

        }

        LIGHTNING_HOST_DEVICE const T& operator()(const int64_t i) const {
            return this->get({i});
        }

        LIGHTNING_HOST_DEVICE T& operator()(const int64_t i) {
            return this->get({i});
        }
};

template <typename T, bool BoundsCheck>
struct Array<T, 2, BoundsCheck>: ArrayBase<T, 2, BoundsCheck> {
    public:
        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<2> strides, const Point<2> lbnd, const Point<2> ubnd) noexcept:
            ArrayBase<T, 2, BoundsCheck>(ptr, strides, lbnd, ubnd) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<2> strides, const Point<2> ubnd) noexcept:
            ArrayBase<T, 2, BoundsCheck>(ptr, strides, ubnd) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<2> strides) noexcept:
            ArrayBase<T, 2, BoundsCheck>(ptr, strides) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, int64_t rows, int64_t cols) noexcept:
            Array(ptr, rows, cols, rows) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, int64_t rows, int64_t cols, int64_t stride) noexcept:
            Array(ptr, {stride, 1}, {0, 0}, {rows, cols}) {
            //
        }

        LIGHTNING_HOST_DEVICE const T& operator()(const int64_t i, const int64_t j) const {
            return this->get({i, j});
        }

        LIGHTNING_HOST_DEVICE T& operator()(const int64_t i, const size_t j) {
            return this->get({i, j});
        }

        LIGHTNING_HOST_DEVICE const Array<T, 1, BoundsCheck> row(const int64_t i) const {
            return this->template collapse_axis<0>(i);
        }

        LIGHTNING_HOST_DEVICE Array<T, 1, BoundsCheck> row(const int64_t i) {
            return this->template collapse_axis<0>(i);
        }

        LIGHTNING_HOST_DEVICE const Array<T, 1, BoundsCheck> col(const int64_t j) const {
            return this->template collapse_axis<1>(j);
        }

        LIGHTNING_HOST_DEVICE Array<T, 1, BoundsCheck> col(const int64_t j) {
            return this->template collapse_axis<1>(j);
        }
};

template <typename T, bool BoundsCheck>
struct Array<T, 3, BoundsCheck>: ArrayBase<T, 3, BoundsCheck> {
    public:
        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<3> strides, const Point<3> lbnd, const Point<3> ubnd) noexcept:
            ArrayBase<T, 3, BoundsCheck>(ptr, strides, lbnd, ubnd) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<3> strides, const Point<3> ubnd) noexcept:
            ArrayBase<T, 3, BoundsCheck>(ptr, strides, ubnd) {
            //
        }

        LIGHTNING_HOST_DEVICE Array(T *const ptr, const Strides<3> strides) noexcept:
            ArrayBase<T, 3, BoundsCheck>(ptr, strides) {
            //
        }

        LIGHTNING_HOST_DEVICE const T& operator()(const int64_t i, const int64_t j, const int64_t k) const {
            return this->get({i, j, k});
        }

        LIGHTNING_HOST_DEVICE T& operator()(const int64_t i, const int64_t j, const int64_t k) {
            return this->get({i, j, k});
        }

        LIGHTNING_HOST_DEVICE const Array<T, 2, BoundsCheck> layer(const int64_t i) const {
            return this->template collapse_axis<0>(i);
        }

        LIGHTNING_HOST_DEVICE Array<T, 2, BoundsCheck> layer(const size_t i) {
            return this->template collapse_axis<0>(i);
        }
};

LIGHTNING_HOST_DEVICE void panic_out_of_bounds() {
#ifdef __CUDA_ARCH__
    assert(!"index out of bounds\n");
    asm("trap;");
    while (1); // helps compiler notice function can never return beyond this point.
#else
    throw std::out_of_range("index out of bounds");
#endif
}

template <size_t N>
LIGHTNING_HOST_DEVICE_NOINLINE void panic_out_of_bounds(Point<N> indices, Point<N> lbnd, Point<N> ubnd) {
    panic_out_of_bounds();
}

template <>
LIGHTNING_HOST_DEVICE_NOINLINE void panic_out_of_bounds<1>(Point<1> indices, Point<1> lbnd, Point<1> ubnd) {
    printf("index %ld out of bounds (must be in %ld...%ld)\n",
            indices[0], lbnd[0], ubnd[0]);
    panic_out_of_bounds();
}

template <>
LIGHTNING_HOST_DEVICE_NOINLINE void panic_out_of_bounds<2>(Point<2> indices, Point<2> lbnd, Point<2> ubnd) {
    printf("index [%ld, %ld] out of bounds (must be between [%ld, %ld] and [%ld, %ld])\n",
            indices[0], indices[1], lbnd[0], lbnd[1], ubnd[0], ubnd[1]);
    panic_out_of_bounds();
}

template <>
LIGHTNING_HOST_DEVICE_NOINLINE void panic_out_of_bounds<3>(Point<3> indices, Point<3> lbnd, Point<3> ubnd) {
    printf("index [%ld, %ld, %ld] out of bounds (must be between [%ld, %ld, %ld] and [%ld, %ld, %ld])\n",
            indices[0], indices[1], indices[2], lbnd[0], lbnd[1], lbnd[2], ubnd[0], ubnd[1], ubnd[2]);
    panic_out_of_bounds();
}


// Some nice aliases.
template <typename T> using Array0 = Array<T, 0>;
template <typename T> using Array1 = Array<T, 1>;
template <typename T> using Array2 = Array<T, 2>;
template <typename T> using Array3 = Array<T, 3>;
template <typename T> using Array4 = Array<T, 4>;
template <typename T> using Array5 = Array<T, 5>;
template <typename T> using Array6 = Array<T, 6>;

template <typename T> using Scalar = Array0<T>;
template <typename T> using Vector = Array1<T>;
template <typename T> using Matrix = Array2<T>;
template <typename T> using Tensor = Array3<T>;


// Some static checks.
static_assert(sizeof(char) == 1, "char must be 8 bit");
static_assert(sizeof(short) == 2, "short must be 16 bit");
static_assert(sizeof(int) == 4, "int must be 32 bit");
static_assert(sizeof(long) == 8, "long must be 64 bit");
static_assert(sizeof(size_t) == 8, "size_t must be 64 bit");

}

#endif //LIGHTNING_H
