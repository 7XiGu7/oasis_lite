-- This is the pattern definition of the interactive table
CREATE TABLE interaction (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    post_id INTEGER,
    comment_id INTEGER,
    original_user_id INTEGER,
    is_mention INTEGER,
    interaction_type TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(post_id) REFERENCES post(post_id),
    FOREIGN KEY(comment_id) REFERENCES comment(comment_id),
    FOREIGN KEY(original_user_id) REFERENCES user(user_id)
);