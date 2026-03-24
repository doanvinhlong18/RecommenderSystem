import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, normalize


class ContentFeatureBuilder:
    """
    Build structured + text content features for hybrid recommendation.

    Final vector = hstack(struct * w_struct, member * w_member, sbert * w_text)
    Không normalize lại sau hstack — để norm của mỗi phần
    bằng đúng weight tương ứng, bất kể số chiều.

    Usage:
        builder = ContentFeatureBuilder()
        builder.fit(df)
        final_matrix = builder.transform(df, sbert_embeddings)
    """

    def __init__(self):
        self.type_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self.genre_list = None
        self.genre_index = None
        self.min_year = None
        self.max_year = None
        self.median_year = None
        self.max_members = None
        self.max_log_episodes = None  # log-normalize episodes

        self._is_fitted = False

    # =====================================================
    # FIT
    # =====================================================
    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # ---------- TYPE ----------
        self.type_encoder.fit(df[["Type"]].fillna("Unknown"))

        # ---------- GENRES ----------
        df["Genres"] = df["Genres"].fillna("").str.lower()
        split_genres = df["Genres"].str.split(",")
        all_genres = {g.strip() for genres in split_genres for g in genres if g.strip()}
        self.genre_list = sorted(all_genres)
        self.genre_index = {g: i for i, g in enumerate(self.genre_list)}

        # ---------- YEAR ----------
        # Chỉ lấy valid years > 0 để tránh min_year = 0
        years = pd.to_numeric(df["year"], errors="coerce")
        valid_years = years[years > 0]
        self.min_year = float(valid_years.min()) if not valid_years.empty else 1960.0
        self.max_year = float(valid_years.max()) if not valid_years.empty else 2024.0
        # median làm giá trị trung tính cho anime thiếu năm
        self.median_year = (
            float(valid_years.median())
            if not valid_years.empty
            else (self.min_year + self.max_year) / 2
        )

        # ---------- EPISODES ----------
        # Log-normalize thay vì categorical bucket:
        # episode count là ordinal continuous — log1p giữ thứ tự tự nhiên
        # và giảm skew của series siêu dài (One Piece 1000+ tập vs 12 tập)
        episodes = pd.to_numeric(df["Episodes"], errors="coerce").fillna(0)
        self.max_log_episodes = float(np.log1p(episodes).max()) or 1.0

        # ---------- MEMBERS ----------
        if "Members" in df.columns:
            members = pd.to_numeric(df["Members"], errors="coerce").fillna(0)
            self.max_members = float(np.log1p(members).max()) or 1.0
        else:
            self.max_members = None

        self._is_fitted = True
        return self

    # =====================================================
    # TRANSFORM
    # =====================================================
    def transform(
        self,
        df: pd.DataFrame,
        sbert_embeddings: np.ndarray,
        w_struct: float = 0.25,
        w_text: float = 0.5,
        w_member: float = 0.25,
    ) -> np.ndarray:
        """
        Build final embedding matrix.

        Weight strategy:
            struct dim ~80, sbert dim 384 → nếu chỉ concat rồi normalize,
            sbert tự nhiên dominate vì nhiều chiều hơn ~5x.

            FIX: normalize từng phần riêng về unit norm TRƯỚC khi nhân weight,
            sau đó hstack mà KHÔNG normalize lại.
            → norm(part_struct) = w_struct
            → norm(part_member) = w_member
            → norm(part_sbert)  = w_text
            → mỗi phần đóng góp đúng theo weight vào cosine similarity,
              bất kể chênh lệch số chiều.

        Default: w_struct=0.25, w_text=0.5, w_member=0.25
            SBERT capture semantic tốt hơn (isekai, dark fantasy, romance...)
            Struct giúp lọc hard constraints: type, genre.
            Member giúp cân bằng theo độ phổ biến và thời đại.
        """
        if not self._is_fitted:
            raise ValueError("ContentFeatureBuilder not fitted. Call fit() first.")

        # FIX: dùng atol rõ ràng để tránh false positive từ float arithmetic
        if not np.isclose(w_struct + w_text + w_member, 1.0, atol=1e-5):
            raise ValueError(
                f"w_struct + w_text + w_member must equal 1.0 "
                f"(got {w_struct + w_text + w_member:.6f})"
            )

        if len(df) != len(sbert_embeddings):
            raise ValueError("df and sbert_embeddings length mismatch")

        df = df.copy()

        # ---------- TYPE ----------
        type_vec = self.type_encoder.transform(df[["Type"]].fillna("Unknown")).astype(
            np.float32
        )

        # ---------- GENRES ----------
        df["Genres"] = df["Genres"].fillna("").str.lower()
        split_genres = df["Genres"].str.split(",")

        genre_matrix = np.zeros((len(df), len(self.genre_list)), dtype=np.float32)
        for i, genres in enumerate(split_genres):
            for g in genres:
                g = g.strip()
                if g in self.genre_index:
                    genre_matrix[i, self.genre_index[g]] = 1.0

        # Normalize genre_matrix riêng trước khi concat vào structured
        # anime nhiều genres có L2 norm cao hơn anime ít genres → sẽ dominate
        # nếu không pre-normalize
        genre_matrix = normalize(genre_matrix)

        # ---------- YEAR ----------
        # fillna bằng median_year (giá trị trung tính)
        # thay vì min_year có thể = 0 → anime thiếu năm bị gán thấp nhất
        years = pd.to_numeric(df["year"], errors="coerce").fillna(self.median_year)
        if self.max_year > self.min_year:
            year_norm = (
                ((years - self.min_year) / (self.max_year - self.min_year))
                .values.reshape(-1, 1)
                .astype(np.float32)
            )
        else:
            year_norm = np.full((len(df), 1), 0.5, dtype=np.float32)

        # ---------- EPISODES ----------
        # Log-normalize: liên tục, tránh ranh giới cứng của categorical bucket
        episodes = pd.to_numeric(df["Episodes"], errors="coerce").fillna(0).values
        ep_norm = (
            (np.log1p(episodes) / self.max_log_episodes)
            .reshape(-1, 1)
            .astype(np.float32)
        )
        ep_norm = np.clip(ep_norm, 0, 1)

        # ---------- MEMBERS ----------
        if self.max_members is not None and "Members" in df.columns:
            members = pd.to_numeric(df["Members"], errors="coerce").fillna(0).values
            members_norm = (
                (np.log1p(members) / self.max_members).reshape(-1, 1).astype(np.float32)
            )
            members_norm = np.clip(members_norm, 0, 1)
        else:
            members_norm = np.zeros((len(df), 1), dtype=np.float32)

        # ---------- CONCAT + NORMALIZE STRUCTURED → unit norm ----------
        structured = np.hstack([type_vec, genre_matrix]).astype(np.float32)
        structured = normalize(structured)

        # FIX: thêm epsilon trước normalize để tránh zero vector khi toàn bộ
        # year/ep/members đều bằng 0 cho một hàng (mất đóng góp cosine)
        member = np.hstack([year_norm, ep_norm, members_norm]).astype(np.float32)
        member = normalize(member + 1e-8)

        # ---------- NORMALIZE SBERT → unit norm ----------
        sbert_norm = normalize(sbert_embeddings.astype(np.float32))

        # ---------- WEIGHTED FUSION ----------
        # normalize từng phần riêng trước → scale với weight sau
        # KHÔNG normalize lại sau hstack — sẽ phá vỡ tỉ lệ weight
        final_vector = np.hstack(
            [
                structured * w_struct,  # norm = w_struct
                member * w_member,  # norm = w_member
                sbert_norm * w_text,  # norm = w_text
            ]
        )

        return final_vector.astype(np.float32)
