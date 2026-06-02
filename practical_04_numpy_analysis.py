import numpy as np

np.random.seed(42)

# ── 1. Array Creation ───────────────────────────────────────────────────────
print("=" * 60)
print("1. Array Creation")
print("=" * 60)
a1 = np.array([10, 20, 30, 40, 50])
a2 = np.arange(1, 11)
a3 = np.linspace(0, 1, 5)
a4 = np.zeros((3, 3))
a5 = np.ones((2, 4))
a6 = np.eye(4)
a7 = np.random.randint(1, 100, (4, 5))
print("1-D array         :", a1)
print("arange            :", a2)
print("linspace          :", a3)
print("zeros (3×3):\n",      a4)
print("ones  (2×4):\n",      a5)
print("identity (4×4):\n",   a6)
print("random int (4×5):\n", a7)

# ── 2. Array Attributes ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Array Attributes")
print("=" * 60)
arr = np.random.rand(3, 4, 2)
print("Shape  :", arr.shape)
print("ndim   :", arr.ndim)
print("size   :", arr.size)
print("dtype  :", arr.dtype)
print("itemsize (bytes):", arr.itemsize)

# ── 3. Indexing & Slicing ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Indexing & Slicing")
print("=" * 60)
m = np.arange(1, 26).reshape(5, 5)
print("Matrix:\n", m)
print("Element [2,3]  :", m[2, 3])
print("Row 1          :", m[1])
print("Col 2          :", m[:, 2])
print("Sub-matrix [1:3, 1:3]:\n", m[1:3, 1:3])
print("Last 2 rows    :\n", m[-2:])
print("Every other col:", m[:, ::2])

# ── 4. Reshaping & Transposing ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Reshaping & Transposing")
print("=" * 60)
flat = np.arange(1, 13)
reshaped = flat.reshape(3, 4)
print("Original :", flat)
print("Reshaped (3×4):\n", reshaped)
print("Transposed:\n", reshaped.T)
print("Flattened:", reshaped.flatten())
print("Ravel    :", reshaped.ravel())

# ── 5. Arithmetic Operations ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Arithmetic Operations")
print("=" * 60)
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
print("x + y  :", x + y)
print("x * y  :", x * y)
print("y / x  :", y / x)
print("x ** 2 :", x ** 2)
print("sqrt(y):", np.sqrt(y))
print("log(y) :", np.log(y).round(3))

# ── 6. Statistical Functions ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Statistical Analysis")
print("=" * 60)
data = np.random.normal(loc=70, scale=15, size=50).round(2)
print("Data (first 10):", data[:10])
print("Mean    :", np.mean(data).round(4))
print("Median  :", np.median(data))
print("Std Dev :", np.std(data).round(4))
print("Variance:", np.var(data).round(4))
print("Min     :", np.min(data))
print("Max     :", np.max(data))
print("Range   :", np.ptp(data).round(4))
print("25th %ile:", np.percentile(data, 25))
print("75th %ile:", np.percentile(data, 75))
print("IQR     :", (np.percentile(data, 75) - np.percentile(data, 25)).round(4))

# ── 7. Boolean Masking ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Boolean Masking & Fancy Indexing")
print("=" * 60)
scores = np.array([45, 78, 92, 55, 88, 63, 71, 39, 95, 60])
print("Scores:", scores)
print("Passed (≥60)  :", scores[scores >= 60])
print("Failed (<60)  :", scores[scores < 60])
print("Count passed  :", np.sum(scores >= 60))
print("% passed      :", np.mean(scores >= 60) * 100, "%")
# np.where
grade = np.where(scores >= 60, "Pass", "Fail")
print("Grades        :", grade)

# ── 8. Matrix Operations ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Matrix Operations (Linear Algebra)")
print("=" * 60)
A = np.array([[2, 1, 3],
              [4, 3, 2],
              [1, 2, 4]])
B = np.array([[1, 0, 2],
              [3, 1, 0],
              [2, 4, 1]])
print("A:\n", A)
print("B:\n", B)
print("A @ B (dot product):\n", A @ B)
print("Determinant of A:", round(np.linalg.det(A), 4))
print("Trace of A      :", np.trace(A))
print("Rank of A       :", np.linalg.matrix_rank(A))
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues     :", eigenvalues.round(4))
try:
    inv_A = np.linalg.inv(A)
    print("Inverse of A:\n", inv_A.round(4))
except np.linalg.LinAlgError:
    print("Matrix A is singular – no inverse.")

# ── 9. Sorting & Searching ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Sorting & Searching")
print("=" * 60)
arr = np.array([64, 25, 12, 22, 11, 90, 45])
print("Original :", arr)
print("Sorted   :", np.sort(arr))
print("Argsort  :", np.argsort(arr))         # indices of sorted elements
print("Max index:", np.argmax(arr))
print("Min index:", np.argmin(arr))
print("Where >40:", np.where(arr > 40))

# ── 10. Stacking & Splitting ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. Stacking & Splitting")
print("=" * 60)
r1 = np.array([1, 2, 3])
r2 = np.array([4, 5, 6])
print("vstack:\n", np.vstack([r1, r2]))
print("hstack :", np.hstack([r1, r2]))
print("column_stack:\n", np.column_stack([r1, r2]))
mat = np.arange(1, 13).reshape(4, 3)
parts = np.vsplit(mat, 2)
print("vsplit parts:", [p.tolist() for p in parts])

# ── 11. Broadcasting ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Broadcasting")
print("=" * 60)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_bias = np.array([10, 20, 30])   # shape (3,) broadcasts over (3,3)
print("Matrix + row_bias:\n", matrix + row_bias)
col_bias = np.array([[1], [2], [3]])
print("Matrix + col_bias:\n", matrix + col_bias)

# ── 12. Random Number Generation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. Random Number Generation")
print("=" * 60)
print("Uniform [0,1)  :", np.random.rand(5).round(4))
print("Normal(0,1)    :", np.random.randn(5).round(4))
print("Integers [1,10):", np.random.randint(1, 10, 8))
print("Choice         :", np.random.choice(["A","B","C"], 6))
sample = np.arange(1, 11)
np.random.shuffle(sample)
print("Shuffled       :", sample)

# ── 13. Practical Mini Analysis ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. Mini Data Analysis – Student Marks")
print("=" * 60)
marks = np.random.randint(30, 100, (20, 5))  # 20 students, 5 subjects
subjects  = ["Math", "Science", "English", "History", "CS"]
totals    = marks.sum(axis=1)
averages  = marks.mean(axis=1).round(2)
toppers   = np.argsort(totals)[::-1][:3]     # top 3 students
sub_avg   = marks.mean(axis=0).round(2)

print("Marks Matrix (first 5 rows):\n", marks[:5])
print("\nStudent Totals (first 5):", totals[:5])
print("Student Averages (first 5):", averages[:5])
print("Top 3 student indices:", toppers)
print("Subject-wise averages:")
for s, a in zip(subjects, sub_avg):
    print(f"  {s}: {a}")
