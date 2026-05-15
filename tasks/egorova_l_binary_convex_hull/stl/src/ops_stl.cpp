#include "egorova_l_binary_convex_hull/stl/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <queue>
#include <thread>
#include <vector>

#include "egorova_l_binary_convex_hull/common/include/common.hpp"
#include "util/include/util.hpp"

namespace egorova_l_binary_convex_hull {

BinaryConvexHullSTL::BinaryConvexHullSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BinaryConvexHullSTL::ValidationImpl() {
  const auto &img = GetInput();
  if (img.width <= 0 || img.height <= 0) {
    return false;
  }
  return static_cast<int>(img.data.size()) == img.width * img.height;
}

bool BinaryConvexHullSTL::PreProcessingImpl() {
  components_ = FindComponents(GetInput());
  return true;
}

bool BinaryConvexHullSTL::RunImpl() {
  const int n = static_cast<int>(components_.size());
  if (n == 0) {
    GetOutput().clear();
    return true;
  }

  const int num_threads = std::min(ppc::util::GetNumThreads(), n);
  std::vector<std::vector<Point>> result(n);
  std::atomic<int> next_idx{0};

  auto worker = [&]() {
    for (;;) {
      const int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
      if (idx >= n) {
        break;
      }
      result[idx] = ConvexHull(components_[idx]);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto &th : threads) {
    th.join();
  }

  GetOutput().clear();
  for (auto &hull : result) {
    if (!hull.empty()) {
      GetOutput().push_back(std::move(hull));
    }
  }
  return true;
}

bool BinaryConvexHullSTL::PostProcessingImpl() {
  return true;
}

std::vector<std::vector<Point>> BinaryConvexHullSTL::FindComponents(const ImageData &img) {
  const int w = img.width;
  const int h = img.height;
  std::vector<bool> visited(static_cast<size_t>(w) * static_cast<size_t>(h), false);
  std::vector<std::vector<Point>> components;

  static const int dx[] = {1, -1, 0, 0};
  static const int dy[] = {0, 0, 1, -1};

  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const size_t idx = static_cast<size_t>(row) * static_cast<size_t>(w) + static_cast<size_t>(col);
      if (img.data[idx] == 0 || visited[idx]) {
        continue;
      }

      std::vector<Point> component;
      std::queue<Point> q;
      q.push({col, row});
      visited[idx] = true;

      while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();
        component.push_back({cx, cy});

        for (int d = 0; d < 4; ++d) {
          const int nx = cx + dx[d];
          const int ny = cy + dy[d];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
            continue;
          }
          const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx);
          if (img.data[nidx] == 0 || visited[nidx]) {
            continue;
          }
          visited[nidx] = true;
          q.push({nx, ny});
        }
      }
      components.push_back(std::move(component));
    }
  }
  return components;
}

static long long Cross(const Point &O, const Point &A, const Point &B) {
  return static_cast<long long>(A.x - O.x) * static_cast<long long>(B.y - O.y) -
         static_cast<long long>(A.y - O.y) * static_cast<long long>(B.x - O.x);
}

std::vector<Point> BinaryConvexHullSTL::ConvexHull(std::vector<Point> points) {
  const int n = static_cast<int>(points.size());
  if (n == 0) {
    return {};
  }
  if (n == 1) {
    return points;
  }

  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  const int m = static_cast<int>(points.size());

  if (m == 1) {
    return points;
  }

  bool collinear = true;
  for (int i = 2; i < m && collinear; ++i) {
    if (Cross(points[0], points[1], points[i]) != 0) {
      collinear = false;
    }
  }
  if (collinear) {
    return {points.front(), points.back()};
  }

  std::vector<Point> hull;
  hull.reserve(2 * m);

  for (int i = 0; i < m; ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }

  const int lower_size = static_cast<int>(hull.size()) + 1;
  for (int i = m - 2; i >= 0; --i) {
    while (static_cast<int>(hull.size()) >= lower_size &&
           Cross(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }

  hull.pop_back();
  return hull;
}

}  // namespace egorova_l_binary_convex_hull
