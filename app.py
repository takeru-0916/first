import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time

# タイトルとアプリ説明
st.title("MediaPipe姿勢解析アプリ")
st.markdown("スマートフォンの動画から姿勢情報・骨格情報を解析するアプリです")

# MediaPipeのPoseモジュールを初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 基本オプション設定
st.sidebar.header("基本設定")
detection_confidence = st.sidebar.slider("検出信頼度", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
tracking_confidence = st.sidebar.slider("トラッキング信頼度", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
show_landmarks = st.sidebar.checkbox("骨格ポイントを表示", value=True)
show_connections = st.sidebar.checkbox("骨格ラインを表示", value=True)

# 品質設定
quality_option = st.sidebar.select_slider(
    "データサイズ",
    options=["高品質", "標準", "小さく"],
    value="標準"
)

# 品質設定に基づくパラメータを設定
quality_params = {
    "高品質": {"scale": 1.0, "sample_rate": 1},
    "標準": {"scale": 0.5, "sample_rate": 5},
    "小さく": {"scale": 0.3, "sample_rate": 10}
}

# 現在の品質設定を取得
current_quality = quality_params[quality_option]
scale_factor = current_quality["scale"]
sample_rate = current_quality["sample_rate"]

# FFmpegの有無を確認
has_ffmpeg = False
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    has_ffmpeg = (result.returncode == 0)
except:
    pass

# 解析結果の保存先を作成する関数
def create_output_folder():
    # 結果保存用のルートディレクトリを作成
    root_result_dir = "result"
    if not os.path.exists(root_result_dir):
        os.makedirs(root_result_dir)
    
    # 日付ディレクトリを作成
    today = datetime.now().strftime("%Y%m%d")
    date_dir = os.path.join(root_result_dir, today)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    
    # 個別の解析結果用ディレクトリを作成
    timestamp = datetime.now().strftime("%H%M%S")
    folder_name = os.path.join(date_dir, f"pose_analysis_{timestamp}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    return folder_name

# 元の動画を保存する関数
def save_original_video(video_file, output_folder):
    original_filename = video_file.name
    original_path = os.path.join(output_folder, original_filename)
    with open(original_path, "wb") as f:
        video_file.seek(0)
        f.write(video_file.read())
    return original_path

# 姿勢解析を実行する関数
def process_video(video_file, output_folder):
    # 一時ファイルを作成して動画を保存
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.seek(0)
    temp_file.write(video_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    
    # 元の動画をフォルダに保存
    original_video_path = save_original_video(video_file, output_folder)
    
    # 動画を開く
    cap = cv2.VideoCapture(temp_file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 回転メタデータを抽出（FFmpegがある場合）
    rotation_metadata = None
    if has_ffmpeg:
        try:
            import subprocess
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                '-show_entries', 'stream_tags=rotate', '-of', 'default=nw=1:nk=1',
                temp_file_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stdout.strip():
                rotation_metadata = result.stdout.strip()
        except:
            pass
    
    # 出力動画の設定
    temp_output_path = os.path.join(output_folder, "temp_analyzed_video.mp4")
    output_video_path = os.path.join(output_folder, "analyzed_video.mp4")
    
    # 回転が必要かどうかを判断
    need_rotation = rotation_metadata in ['90', '270']
    
    # 出力サイズを調整
    output_width = int(width * scale_factor) if not need_rotation else int(height * scale_factor)
    output_height = int(height * scale_factor) if not need_rotation else int(width * scale_factor)
    
    # 幅と高さを偶数に調整
    output_width = output_width if output_width % 2 == 0 else output_width + 1
    output_height = output_height if output_height % 2 == 0 else output_height + 1
    
    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (output_width, output_height))
    
    # 姿勢解析のためのMediaPipe Poseインスタンスを作成
    with mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as pose:
        # 進捗バーを作成
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 解析開始時間
        start_time = time.time()
        
        # 関節データを保存するリスト
        joint_data = []
        
        # フレームカウンター
        frame_idx = 0
        
        # 各フレームを処理
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            
            # MediaPipeはRGBを使用するのでBGR->RGBに変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 姿勢検出を実行
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            
            # 検出結果を描画
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # リサイズして処理を軽くする
            image_bgr = cv2.resize(image_bgr, (output_width, output_height))
            
            # 姿勢ランドマークがある場合
            if results.pose_landmarks:
                # 骨格ポイントと接続線を描画
                if show_landmarks and show_connections:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                elif show_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                elif show_connections:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style()
                    )
                
                # 関節データを収集
                frame_time = frame_idx / fps
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    joint_data.append({
                        'frame': frame_idx,
                        'time': frame_time,
                        'joint_id': idx,
                        'joint_name': mp_pose.PoseLandmark(idx).name,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
            
            # 回転が必要な場合、フレームを回転
            if need_rotation:
                if rotation_metadata == '90':
                    # 90度時計回りに回転
                    image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_metadata == '270':
                    # 90度反時計回りに回転
                    image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # 処理したフレームを出力動画に書き込む
            try:
                out.write(image_bgr)
            except Exception as e:
                st.error(f"フレーム書き込みエラー: {e}")
                break
            
            # 進捗状況を更新
            progress = (frame_idx + 1) / frame_count
            progress_bar.progress(progress)
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            status_text.text(f"処理中: {frame_idx+1}/{frame_count} フレーム " +
                           f"({progress*100:.1f}%) | 残り時間: {remaining_time:.1f}秒")
            
            frame_idx += 1
        
        # リソースを解放
        cap.release()
        out.release()
        
        # 動画の最終処理
        if has_ffmpeg:
            try:
                import subprocess
                # 高圧縮で出力
                cmd = [
                    "ffmpeg", "-y", "-i", temp_output_path,
                    "-c:v", "libx264", "-crf", "28",
                    "-preset", "faster", "-c:a", "aac",
                    output_video_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                os.remove(temp_output_path)
            except:
                import shutil
                shutil.move(temp_output_path, output_video_path)
        else:
            import shutil
            shutil.move(temp_output_path, output_video_path)
        
        # 一時ファイルを削除
        os.unlink(temp_file_path)
        
        # CSVファイルに関節データを保存
        if joint_data:
            # データフレームを作成
            raw_df = pd.DataFrame(joint_data)
            
            # フレームを間引き
            frame_numbers = raw_df['frame'].unique()
            sampled_frames = [f for i, f in enumerate(frame_numbers) if i % sample_rate == 0]
            sampled_df = raw_df[raw_df['frame'].isin(sampled_frames)]
            
            # フレームごとにピボットする
            pivot_df = pd.DataFrame()
            for frame in sampled_df['frame'].unique():
                frame_data = sampled_df[sampled_df['frame'] == frame]
                row_data = {'frame': frame, 'time': frame_data['time'].iloc[0]}
                
                # 各関節のデータを列に変換
                for _, row in frame_data.iterrows():
                    joint_name = row['joint_name']
                    row_data[f"{joint_name}_x"] = row['x']
                    row_data[f"{joint_name}_y"] = row['y']
                    if quality_option == "高品質":
                        row_data[f"{joint_name}_z"] = row['z']
                        row_data[f"{joint_name}_visibility"] = row['visibility']
                
                pivot_df = pd.concat([pivot_df, pd.DataFrame([row_data])], ignore_index=True)
            
            # CSVファイルに保存
            csv_path = os.path.join(output_folder, "joint_data.csv")
            pivot_df.to_csv(csv_path, index=False, float_format='%.4f')
        
        return output_video_path, os.path.join(output_folder, "joint_data.csv"), frame_idx, len(joint_data), original_video_path

# ファイルアップローダーを作成
uploaded_file = st.file_uploader("スマートフォンの動画をアップロード", type=['mp4', 'mov', 'avi'])

# メインアプリの実行ロジック
if uploaded_file is not None:
    st.video(uploaded_file)
    
    # セッション状態を確認して、新しいファイルがアップロードされた場合のみ解析を実行
    current_file = uploaded_file.name + str(uploaded_file.size)
    
    if 'last_processed_file' not in st.session_state:
        st.session_state.last_processed_file = None
    
    # 新しいファイルがアップロードされたか確認
    if st.session_state.last_processed_file != current_file:
        st.session_state.last_processed_file = current_file
        
        # 自動的に解析を実行
        with st.spinner("動画を解析中..."):
            try:
                # 出力フォルダを作成
                output_folder = create_output_folder()
                
                # 動画を処理
                output_video, output_csv, frame_count, landmark_count, original_video_path = process_video(uploaded_file, output_folder)
                
                # 結果を表示
                st.success(f"解析完了！ {frame_count}フレームを処理しました。")
                
                # 処理された動画を表示
                st.subheader("骨格情報を含む解析済み動画")
                st.video(output_video)
                
                # データ概要を表示
                st.subheader("解析データの概要")
                st.write(f"検出された骨格ポイント数: {landmark_count}")
                st.write(f"出力フォルダ: {os.path.abspath(output_folder)}")
                
                # CSVデータをプレビュー
                if os.path.exists(output_csv):
                    st.subheader("関節位置データ (先頭10行)")
                    data = pd.read_csv(output_csv)
                    st.dataframe(data.head(10))
                    
                    # ダウンロードリンク
                    with open(output_csv, 'rb') as file:
                        st.download_button(
                            label="関節データCSVをダウンロード",
                            data=file,
                            file_name="joint_data.csv",
                            mime="text/csv"
                        )
                
                # 動画ダウンロードリンク
                with open(output_video, 'rb') as file:
                    st.download_button(
                        label="解析済み動画をダウンロード",
                        data=file,
                        file_name="analyzed_video.mp4",
                        mime="video/mp4"
                    )
                    
                # 出力フォルダの情報
                st.info(f"全ての解析データは次のフォルダに保存されています: {os.path.abspath(output_folder)}")
                
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.info("動画ファイルをアップロードしてください")
    
# フッター
st.markdown("---")
st.caption("© 2025 MediaPipe姿勢解析アプリ")