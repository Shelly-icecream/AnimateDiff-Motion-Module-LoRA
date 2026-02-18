import yt_dlp
import os


def is_good_video(info):
    if info is None:
        return False

    title = (info.get("title") or "").lower()
    desc = (info.get("description") or "").lower()
    text = title + " " + desc

    must_keywords = [
        "slow motion", "slow-mo", "slomo",
        "240fps", "480fps", "960fps", "1000fps",
        "high speed", "high-speed"
    ]
    if not any(k in text for k in must_keywords):
        return False

    ban_keywords = [
        "ai", "generated", "stable diffusion", "midjourney",
        "runway", "pika", "sora", "luma", "kaiber"
    ]
    if any(k in text for k in ban_keywords):
        return False

    ban_keywords2 = [
        "tutorial", "how to", "settings", "test",
        "review", "comparison", "vs"
    ]
    if any(k in text for k in ban_keywords2):
        return False

    return True


def search_videos(query, n=200):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,  # 只拿列表，不下载
        "skip_download": True,
    }
    search_query = f"ytsearch{n}:{query}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(search_query, download=False)
        entries = result.get("entries", [])
        urls = []
        for e in entries:
            if e and "url" in e:
                urls.append(e["url"])
        return urls


def download_one(url, folder):
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "ignoreerrors": True,
        "noplaylist": True,
        "match_filter": yt_dlp.utils.match_filter_func("duration <= 30"),
        "outtmpl": f"{folder}/%(title)s.%(ext)s",
        "quiet": False,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not is_good_video(info):
            print(f"[skip] {info.get('title','(no title)')}")
            return False

        ydl.download([url])
        return True


def download_youtube_videos(tasks, max_per_query):
    for folder, query in tasks.items():
        os.makedirs(folder, exist_ok=True)

        print("\n" + "=" * 40)
        print(f"主题: {folder}")
        print(f"关键词: {query}")
        print("=" * 40)

        urls = search_videos(query, n=300)

        ok = 0
        for url in urls:
            if ok >= max_per_query:
                break
            try:
                if download_one(url, folder):
                    ok += 1
                    print(f"✅ 已下载 {ok}/{max_per_query}")
            except Exception as e:
                print(f"[error] {e}")

        print(f"🎉 {folder} 完成: {ok}/{max_per_query}")


if __name__ == "__main__":
    tasks = {
        #"SlowMotion_240fps": "slow motion 240fps walking",
        #"SlowMotion_Action": "slow motion 240fps running",
        "SlowMotion_HairCloth": "slow motion 240fps hair flip",
    }

    download_youtube_videos(tasks, max_per_query=50)
