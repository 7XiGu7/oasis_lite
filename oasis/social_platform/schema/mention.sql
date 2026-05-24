-- This is the schema definition for the mention table
CREATE TABLE mention (
    mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, -- The user who made the mention
    mentioned_user_id INTEGER, -- The user who is mentioned
    post_id INTEGER, -- The post where the mention occurred
    comment_id INTEGER, -- The comment where the mention occurred, NULL if this occurred in a post
    created_at DATETIME,
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(mentioned_user_id) REFERENCES user(user_id),
    FOREIGN KEY(post_id) REFERENCES post(post_id),
    FOREIGN KEY(comment_id) REFERENCES comment(comment_id)
);