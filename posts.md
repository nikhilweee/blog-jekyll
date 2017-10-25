---
title: Posts
layout: page
---

<div class="tags-expo">
  <ul class="pager main-pager">
    
    <li>
    <a href="#" class="post-tag">All what I've written so far</a>
    </li>
    
  </ul>
  <hr/>
  <div class="tags-expo-section">
    <ul class="tags-expo-posts">
      {% for post in site.posts %}
      <li>
        <a class="post-title" href="{{ site.baseurl }}{{ post.url }}">
          <span style="color: #404040;">{{ post.date | date_to_string }}</span>
          <strong>{{ post.title }}</strong><br>{{ post.subtitle }}
        </a>
      </li>
      {% endfor %}
    </ul>
  </div>
</div>
